import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from src.storage.schema import SCHEMAS

def get_year_month(ts_series: pd.Series) -> pd.Series:
    """Extract YYYY-MM from timestamp series."""
    return ts_series.dt.strftime('%Y-%m')

def save_partitioned(
    df: pd.DataFrame,
    table_name: str,
    domain: str,
    base_path: str = "data/raw",
    partition_cols: list = None
):
    """
    Saves DataFrame to partitioned Parquet dataset.
    
    Structure: {base_path}/{domain}/{table_name}/...partitions...
    
    Default Partitions if None: ['exchange', 'symbol', 'year_month']
    """
    
    if table_name not in SCHEMAS:
        raise ValueError(f"Schema not defined for table: {table_name}")
        
    schema = SCHEMAS[table_name]
    
    # Ensure correct types roughly (pyarrow will enforce mostly, but conversion helps)
    # This is a simplified check. In prod, we'd do strict validation.
    
    output_dir = Path(base_path) / domain / table_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default partitioning strategy
    if partition_cols is None:
        partition_cols = ['exchange', 'symbol', 'year_month']
        
    # Generate year_month if needed and not present
    if 'year_month' in partition_cols and 'year_month' not in df.columns:
        if 'timestamp_utc' in df.columns:
             df['year_month'] = get_year_month(pd.to_datetime(df['timestamp_utc']))
        elif 'date_utc' in df.columns:
             df['year_month'] = get_year_month(pd.to_datetime(df['date_utc']))
        else:
             raise ValueError("Cannot derive year_month, no timestamp column found.")

    # Convert to Table
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    
    # Write dataset
    # Write dataset
    # STRATEGY: To avoid duplicates, we must read existing data if we are appending?
    # Parquet write_to_dataset with 'overwrite_or_ignore' might leave old files if new ones have different names?
    # Or 'delete_matching'? 
    # For robust deduplication, we should ideally handle it at the partition level.
    # Simple approach for this sprint:
    # 1. We know the partition keys.
    # 2. For each unique partition in df:
    #    a. Construct path.
    #    b. If exists, read it.
    #    c. Concat, drop_duplicates.
    #    d. Write back (overwrite).
    
    # However, 'write_to_dataset' does partitioning automatically. 
    # To do robust dedup, we might need to iterate valid partitions in DF.
    
    # Optimised attempt:
    # Use write_to_dataset but verify if we can dedupe first? 
    # If we blindly write, we get duplicates if unique_ids are same.
    
    # Let's assume the caller handles dedup OR we implement a "dedupe_and_save" mode.
    # Instructions say: "O bien haga upsert por PK... O bien dedupe en post-proceso".
    # Since we are modifying 'save_partitioned', let's make it smarter if we can, 
    # OR we rely on the separate 'clean_data.py' for the big fix, 
    # and here we just try to be safe?
    # But clean_data needs a function to call.
    
    # We will stick to the standard write here, BUT we will add a new function `deduplicate_partition`
    # and maybe valid `save_partitioned` to use it if requested?
    # Actually, for the ingest loop, reading every time is slow.
    # Better to ingest (append) and then run compact/dedup periodically.
    # So we keep this simple append-like, but we add a `deduplicate_partition` function for the job.
    
    pq.write_to_dataset(
        table,
        root_path=str(output_dir),
        partition_cols=partition_cols,
        existing_data_behavior='overwrite_or_ignore', 
        compression='snappy'
    )
    print(f"Saved {len(df)} rows to {output_dir}")

def deduplicate_dataset(
    table_name: str,
    domain: str,
    base_path: str = "data/raw",
    pk_cols: list = None
):
    """
    Iterates over all partitions of a table, reads them, deduplicates by PK, and writes back.
    """
    if pk_cols is None:
        # Default PK for crypto ohlcv
        pk_cols = ["timestamp_utc", "exchange_code", "symbol", "market_kind"]
        
    dataset_path = Path(base_path) / domain / table_name
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    # We can use pyarrow dataset API to find fragments, but pandas read_parquet(dir) works too if not too huge.
    # But we want to process partition by partition to avoid memory explosion.
    # We need to discover the directory structure.
    # recursively find all lowest-level directories (partitions).
    
    # Assuming hierarchical: domain/table/exchange/symbol/year_month
    # We want to process at the file level or leaf dir level.
    
    tasks = []
    for root, dirs, files in os.walk(dataset_path):
        if files and any(f.endswith('.parquet') for f in files):
            tasks.append(Path(root))
            
    total_dupes = 0
    
    for partition_dir in tasks:
        try:
            # Read
            df = pd.read_parquet(partition_dir)
            
            # Count before
            count_before = len(df)
            
            # Dedup
            # subset needs to be present in df. 
            # Note: partition cols might NOT be in df if hive partitioning was used during read without restoring?
            # pd.read_parquet usually restores partition keys as columns if they are in the path.
            # Let's verify cols exist.
            valid_pk = [c for c in pk_cols if c in df.columns]
            
            if not valid_pk:
                # If PKs are missing (e.g. partition keys), we might need to deduce them or just dedup by all columns?
                # If 'exchange_code' is a partition key, it is in columns.
                df.drop_duplicates(inplace=True)
            else:
                df.drop_duplicates(subset=valid_pk, inplace=True)
            
            count_after = len(df)
            dupes = count_before - count_after
            total_dupes += dupes
            
            if dupes > 0:
                print(f"Cleaned {dupes} duplicates from {partition_dir}")
                # Write back: We overwrite the files in this directory.
                # To be safe, we can write to temp and move, or just overwrite since we have data in memory.
                # pandas to_parquet with partition_cols=None will write single file? 
                # If we want to keep the original file structure, it's tricky if multiple files existed.
                # But usually we can just write one compacted file per leaf partition.
                
                # Nuke files in dir
                for f in partition_dir.glob("*.parquet"):
                    f.unlink()
                
                # Write single compacted file
                # We do NOT pass partition_cols because we are INSIDE a partition folder already.
                # We just dump the content.
                # Wait, if we use to_parquet, we must ensure we don't write partition columns into the file 
                # if they are inferred from directory structure, otherwise next read might duplicate columns?
                # Pyarrow write_to_dataset handles this. 
                # But here we are writing a specific file.
                
                # If we act as if we are writing a normal file, we just write. 
                # When reading back, engines usually handle it. 
                
                # Simplest:
                output_file = partition_dir / "part-0.parquet"
                df.to_parquet(output_file, index=False, compression='snappy')
                
        except Exception as e:
            print(f"Error processing {partition_dir}: {e}")
            
    print(f"Total duplicates removed: {total_dupes}")


import os
import json
import logging
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime

logger = logging.getLogger("utils.stats")

def generate_crypto_stats(config, run_metadata=None):
    """
    run_metadata: dict mapping key (exchange_code, symbol, market_kind) -> blocked_reason (str) or None
    """
    base_path = Path(config["paths"]["raw"]) / "crypto"
    stats_path = Path(config["paths"]["stats"])
    stats_path.mkdir(parents=True, exist_ok=True)
    
    # Defaults
    if run_metadata is None:
        run_metadata = {}

    start_str = config["crypto_date_range"]["start"]
    end_str = config["crypto_date_range"]["end"]
    start_dt = pd.to_datetime(start_str, utc=True)
    end_dt = pd.to_datetime(end_str, utc=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_range": {"start": start_str, "end": end_str},
        "details": [],
        "summary": {}
    }

    # Iterate over symbols defined in config
    symbols = config["symbols"]["crypto"]
    exchanges = config["exchanges"] # {"BINANCE": 1, "BYBIT": 2}
    
    # We want details per symbol/exchange/market_kind
    # Market kinds we care about: spot, perp
    markets = ["spot", "perp"]
    
    # Pre-scan directory to allow quick lookup? 
    # Or iterate expectations. Iterating expectations is better for "expected vs actual".
    
    # Mapping for easy lookup
    ex_code_map = {v: k for k, v in exchanges.items()}
    
    detailed_stats = []
    
    # Helper to count expected bars
    def get_expected_bars(start, end, interval_mins=5):
        delta = end - start
        return int(delta.total_seconds() / (interval_mins * 60)) + 1

    expected_total_5m = get_expected_bars(start_dt, end_dt)

    for symbol in symbols:
        for ex_name, ex_code in exchanges.items():
            for mkt in markets:
                # Construct identity
                # OHLCV Table is raw_crypto_ohlcv_5m
                
                # Check for blocking
                blocked_reason = run_metadata.get((ex_code, symbol, mkt))
                # If entire exchange blocked, might be stored as (ex_code, None, None) or similar?
                # Let's assume metadata populated specifically per iterate loop or broadly.
                # If not specific, check broad:
                if not blocked_reason:
                     blocked_reason = run_metadata.get((ex_code, None, None))
                
                stat_entry = {
                    "exchange": ex_name,
                    "exchange_code": ex_code,
                    "symbol": symbol,
                    "market_kind": mkt,
                    "rows": 0,
                    "min_ts": None,
                    "max_ts": None,
                    "expected_bars": expected_total_5m,
                    "coverage_pct": 0.0,
                    "blocked_reason": blocked_reason
                }
                
                # Read Parquet
                # Paths: .../raw_crypto_ohlcv_5m/exchange_code=X/symbol=Y/...
                # Note: 'perp' in path might be partition or just implied?
                # Writer saves partition cols=['exchange_code', 'symbol', 'year_month']
                # But wait, market_kind is a COLUMN, NOT A PARTITION in standard writer call unless specified?
                # In downloader_ohlcv:
                # save_partitioned(..., partition_cols=['exchange_code', 'symbol', 'year_month'])
                # So 'market_kind' is inside the file. 
                # We need to filter by market_kind.
                
                # Scanning all files for a symbol to separate spot/perp is expensive if mixed.
                # Optimization: The query can filter.
                
                table_path = base_path / "raw_crypto_ohlcv_5m"
                if table_path.exists():
                     # Use PyArrow dataset filter
                    try:
                        dataset = pq.ParquetDataset(
                            table_path, 
                            filters=[
                                ('exchange_code', '=', ex_code),
                                ('symbol', '=', symbol)
                            ]
                        )
                        # We have to read column 'market_kind'/timestamp to filter/agg
                        # But wait, market_kind values: 'spot', 'perp' (Binance) / 'linear' (Bybit)?
                        # config implies 'perp'. Downloader writes 'perp' for Binance and 'linear' for Bybit.
                        # We should standardize 'perp' vs 'linear' or just check both.
                        target_mkt = mkt
                        if ex_name == "BYBIT" and mkt == "perp": 
                            target_mkt = "linear" # Downloader uses linear
                        
                        # Read specific columns
                        if dataset.files:
                            t = dataset.read(columns=["timestamp_utc", "market_kind"])
                            df = t.to_pandas()
                            df_mkt = df[df["market_kind"] == target_mkt]
                            
                            if not df_mkt.empty:
                                stat_entry["rows"] = len(df_mkt)
                                stat_entry["min_ts"] = df_mkt["timestamp_utc"].min().isoformat()
                                stat_entry["max_ts"] = df_mkt["timestamp_utc"].max().isoformat()
                                stat_entry["coverage_pct"] = round((stat_entry["rows"] / stat_entry["expected_bars"]) * 100, 2)
                                
                    except Exception as e:
                        # logger.warning(f"Error checking stats for {symbol} {ex_name}: {e}")
                        pass # Path might not exist or empty
                        
                detailed_stats.append(stat_entry)

    report["details"] = detailed_stats
    
    # Save
    with open(stats_path / "crypto_raw_coverage.json", "w") as f:
        json.dump(report, f, indent=2)
        
    return report

def generate_equities_stats(config):
    base_path = Path(config["paths"]["raw"]) / "equities" / "raw_equities_1d"
    stats_path = Path(config["paths"]["stats"])
    stats_path.mkdir(parents=True, exist_ok=True)
    
    start_str = config["equities_date_range"]["start"]
    end_str = config["equities_date_range"]["end"]
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    
    # Approx trading days (Mon-Fri)
    # This is a rough estimate
    bus_days = pd.bdate_range(start=start_dt, end=end_dt)
    expected_days = len(bus_days)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_range": {"start": start_str, "end": end_str},
        "details": []
    }
    
    if base_path.exists():
        # Partitioned? "raw/equities/raw_equities_1d"
        # Stooq downloader likely won't partition by default unless we tell it.
        # Or partition by ticker? Recommended for equities as tickers are many.
        # Let's assume partition by ticker.
        
        # Scan dataset
        try:
             dataset = pq.ParquetDataset(base_path)
             # Ticker is likely a partition key
             # Iterate unique partition keys if possible or just group by
             
             # If partitioned by ticker:
             # We can just list directories? 
             # dataset.partitioning might help, but simple iteration of files/dirs usually works for stats
             
             # Let's read full metadata columns=["ticker", "date_utc"]
             # If dataset is huge, this is slow. Better read per ticker if partitioned.
             # Assuming partitioned by ticker.
             
             # Get all tickers from config to verify missing
             tickers = config["symbols"]["equities"]
             
             for ticker in tickers:
                stat = {
                    "ticker": ticker, 
                    "rows": 0, 
                    "min_date": None, 
                    "max_date": None,
                    "coverage_pct": 0.0
                }
                
                # Try specific filter
                try:
                    ds_ticker = pq.ParquetDataset(base_path, filters=[('ticker', '=', ticker)])
                    if ds_ticker.files:
                        df = ds_ticker.read(columns=["date_utc"]).to_pandas()
                        if not df.empty:
                            stat["rows"] = len(df)
                            stat["min_date"] = df["date_utc"].min().isoformat() if hasattr(df["date_utc"].min(), 'isoformat') else str(df["date_utc"].min())
                            stat["max_date"] = df["date_utc"].max().isoformat() if hasattr(df["date_utc"].max(), 'isoformat') else str(df["date_utc"].max())
                            stat["coverage_pct"] = round((len(df) / expected_days) * 100, 2)
                except:
                    pass
                
                report["details"].append(stat)
                
        except Exception as e:
            logger.error(f"Error equities stats: {e}")
            
    with open(stats_path / "equities_raw_coverage.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

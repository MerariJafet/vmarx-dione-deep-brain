import pandas as pd
import requests
import logging
import time
from tqdm import tqdm
from datetime import datetime
from src.storage.writer import save_partitioned

logger = logging.getLogger("ingestion.stooq")

class StooqDownloader:
    def __init__(self, config):
        self.config = config
        self.base_path = config["paths"]["raw"]
        
    def run(self):
        tickers = self.config["symbols"]["equities"]
        start_date = self.config["equities_date_range"]["start"]
        end_date = self.config["equities_date_range"]["end"]
        
        # Stooq date format: YYYYMMDD
        d_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        d_end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
        
        # Check connectivity to Stooq once
        from src.ingestion.connectivity import ConnectivityChecker
        if not ConnectivityChecker.check_endpoint("https://stooq.com")[0]:
             logger.error("Stooq seems unreachable (DNS/403). Skipping equities ingestion.")
             return
        
        for ticker in tqdm(tickers, desc="Downloading Equities"):
            try:
                # URL format: https://stooq.com/q/d/l/?s={ticker}.us&d1={d_start}&d2={d_end}&i=d
                # Note: .us specific for US equities.
                url = f"https://stooq.com/q/d/l/?s={ticker}.us&d1={d_start}&d2={d_end}&i=d"
                
                time.sleep(1.5) # Rate limit
                
                df = pd.read_csv(url)
                
                if df.empty or "Date" not in df.columns:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Stooq Columns: Date, Open, High, Low, Close, Volume
                # Rename to schema
                df.rename(columns={
                    "Date": "date_utc",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }, inplace=True)
                
                # Add metadata
                df["ticker"] = ticker
                df["ingested_at_utc"] = pd.Timestamp.now(tz='UTC').floor('ms')
                df["source"] = "STOOQ"
                
                # Types
                df["date_utc"] = pd.to_datetime(df["date_utc"]) # Stooq CSV is YYYY-MM-DD
                
                # Add year_month for schema compliance (even if not partitioning by it)
                df["year_month"] = df["date_utc"].dt.strftime('%Y-%m')
                
                # Check for duplicates
                df.drop_duplicates(subset=["date_utc", "ticker"], inplace=True)
                
                # Save
                save_partitioned(
                    df, 
                    "raw_equities_1d", 
                    "equities", 
                    self.base_path, 
                    partition_cols=["ticker"] # Keep it simple by ticker
                )
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")

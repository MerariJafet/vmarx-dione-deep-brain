import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from src.storage.writer import save_partitioned

logger = logging.getLogger("processing.aligned")

class AlignedLayerBuilder:
    def __init__(self, config):
        self.config = config
        self.raw_path = Path(config["paths"]["raw"])
        self.aligned_path = Path(config["paths"]["aligned"])
        self.aligned_path.mkdir(parents=True, exist_ok=True)
        
    def build_crypto_5m(self):
        """
        Creates global 5m grid for config range and joins with raw data.
        Fills missing values with nan (or explicit logic).
        """
        start_date = self.config["crypto_date_range"]["start"]
        end_date = self.config["crypto_date_range"]["end"]
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        # Grid: 5m = 300000ms
        grid = range(start_ts, end_ts + 1, 300000)
        grid_df = pd.DataFrame({"timestamp_utc": grid})
        grid_df["timestamp_utc"] = pd.to_datetime(grid_df["timestamp_utc"], unit="ms", utc=True)
        
        symbols = self.config["symbols"]["crypto"]
        # Only implementing for Binance Spot as others are blocked/empty?
        # User says: "Crypto SPOT (Binance) con OI/FUND/BASIS=MISS"
        # We iterate symbols.
        
        # We need raw data.
        # We can read strictly the chunks we need or all.
        
        for symbol in symbols:
            # Load Raw Spot
            # Path: raw/crypto/raw_crypto_ohlcv_5m/exchange_code=1/symbol={symbol}
            # We assume exchange_code=1 for binance.
            input_dir = self.raw_path / "crypto" / "raw_crypto_ohlcv_5m" / "exchange_code=1" / f"symbol={symbol}"
            
            if not input_dir.exists():
                logger.warning(f"No raw data for {symbol}, skipping aligned build.")
                continue
                
            try:
                # Read raw
                # We need to dedupe just in case? Or rely on clean_data. 
                # clean_data ran, so we should be good.
                raw_df = pd.read_parquet(input_dir)
                
                # Filter for 'spot' just in case perp is mixed (it shouldn't be if we check file paths but logic allowed mixing)
                raw_df = raw_df[raw_df["market_kind"] == "spot"].copy()
                
                # drop duplicates just in case
                raw_df.drop_duplicates(subset=["timestamp_utc"], inplace=True)
                
                # Align types for merge
                if "timestamp_utc" in raw_df.columns:
                    raw_df["timestamp_utc"] = pd.to_datetime(raw_df["timestamp_utc"], utc=True)
                
                # Merge
                merged = pd.merge(grid_df, raw_df, on="timestamp_utc", how="left")
                
                # Flags
                merged["is_missing"] = merged["close"].isna()
                merged["exchange_code"] = 1
                merged["symbol"] = symbol
                merged["market_kind"] = "spot"
                
                # Columns strict order/selection?
                # User: SPREAD/FLOW -> NULL if not exists.
                # Raw has: open, high, low, close, volume, quote_volume, trade_count
                
                # We do NOT forward fill RET/RVOL/VLM/etc (those are patch level).
                # Aligned layer just holds the grid.
                
                # Save
                # Partition by symbol, year_month
                save_partitioned(
                    merged,
                    "aligned_crypto_5m",
                    "crypto",
                    base_path="data/aligned",
                    partition_cols=["symbol", "year_month"]
                )
                        
            except Exception as e:
                logger.error(f"Error building aligned for {symbol}: {e}")

    def build_equities_1d(self):
        """
        Creates daily grid.
        """
        start_date = self.config["equities_date_range"]["start"]
        end_date = self.config["equities_date_range"]["end"]
        
        # Business days grid? Or just daily?
        # User: "aligned_equities_1d: daily rows, missing_flag si faltan días de trading (NO imputar)"
        # Usually implies filtered to trading days OR full cal. 
        # "Missing stats" implied business days check.
        # Let's use BDay grid.
        
        grid_dt = pd.bdate_range(start=start_date, end=end_date)
        grid_df = pd.DataFrame({"date_utc": grid_dt})
        
        symbols = self.config["symbols"]["equities"]
        
        for ticker in symbols:
            input_dir = self.raw_path / "equities" / "raw_equities_1d" / f"ticker={ticker}"
            
            if not input_dir.exists():
                continue
                
            try:
                raw_df = pd.read_parquet(input_dir)
                # Ensure date_utc is datetime
                raw_df["date_utc"] = pd.to_datetime(raw_df["date_utc"])
                
                merged = pd.merge(grid_df, raw_df, on="date_utc", how="left")
                merged["is_missing"] = merged["close"].isna()
                merged["ticker"] = ticker
                merged["source"] = "STOOQ"
                
                save_partitioned(
                    merged,
                    "aligned_equities_1d",
                    "equities",
                    base_path="data/aligned",
                    partition_cols=["ticker"]
                )
                
            except Exception as e:
                logger.error(f"Error building aligned equities {ticker}: {e}")

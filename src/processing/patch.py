import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.storage.writer import save_partitioned

logger = logging.getLogger("processing.patch")

class PatchLayerBuilder:
    def __init__(self, config):
        self.config = config
        self.aligned_path = Path(config["paths"]["aligned"])
        self.patch_path = Path(config["paths"]["patch"])
        self.patch_path.mkdir(parents=True, exist_ok=True)
        
    def build_crypto_1h(self):
        """
        Resamples aligned 5m to 1h patch.
        RET, RVOL, VLM, FLOW, SPREAD, OI, FUND, BASIS.
        """
        symbols = self.config["symbols"]["crypto"]
        
        for symbol in symbols:
            input_dir = self.aligned_path / "crypto" / "aligned_crypto_5m" / f"symbol={symbol}"
            if not input_dir.exists():
                continue
            
            try:
                df = pd.read_parquet(input_dir)
                df = df.sort_values("timestamp_utc")
                df["datetime"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
                df.set_index("datetime", inplace=True)
                
                # Aggregations
                # We need to group by 1H.
                
                # RET: Log return of the patch (Close_end / Open_start)
                # Note: If Open_start or Close_end is missing, Ret is NaN.
                
                # Helper for first/last
                def first_s(s): return s.iloc[0] if len(s)>0 else np.nan
                def last_s(s): return s.iloc[-1] if len(s)>0 else np.nan
                
                hashed = df.resample("1h").agg({
                    "open": first_s,
                    "close": last_s,
                    "volume": "sum",
                    "timestamp_utc": "first" # patch timestamp
                })
                
                # Calculate intrapatch volatility (RVOL)
                # RVOL = sqrt(sum(r_i^2)) ? or std? 
                # user: "realized vol intrapatch (sobre returns 5m)"
                # Let's calculate 5m log returns first
                df["log_ret_5m"] = np.log(df["close"] / df["open"])
                # Handle zeros/errors? valid close/open required.
                
                # now resample square sum
                df["sq_ret"] = df["log_ret_5m"] ** 2
                rvol_series = df["sq_ret"].resample("1h").sum().map(np.sqrt)
                
                patch_df = hashed.copy()
                patch_df["RVOL"] = rvol_series
                
                # RET
                patch_df["RET"] = np.log(patch_df["close"] / patch_df["open"])
                
                # VLM
                patch_df.rename(columns={"volume": "VLM"}, inplace=True)
                
                # BLOCKED / MISSING
                patch_df["FLOW"] = np.nan
                patch_df["SPREAD"] = np.nan
                patch_df["OI"] = np.nan
                patch_df["FUND"] = np.nan
                patch_df["BASIS"] = np.nan
                
                # Metadata
                patch_df["symbol"] = symbol
                patch_df["exchange_code"] = 1
                patch_df["market_kind"] = "spot"
                
                # Payload Order Fix: [RET, RVOL, VLM, FLOW, SPREAD, OI, FUND, BASIS]
                patch_df = patch_df[[
                    "timestamp_utc", "symbol", "RET", "RVOL", "VLM", 
                    "FLOW", "SPREAD", "OI", "FUND", "BASIS",
                    "exchange_code", "market_kind"
                ]]
                
                # Save
                save_partitioned(
                    patch_df,
                    "patch_crypto_1h",
                    "patch",
                    base_path="data", # saves to data/patch/patch_crypto_1h
                    partition_cols=["symbol"]
                )
                
            except Exception as e:
                logger.error(f"Error building patch crypto {symbol}: {e}")

    def build_equities_5d(self):
        """
        Resamples aligned 1d to 5d patch.
        """
        symbols = self.config["symbols"]["equities"]
        
        for ticker in symbols:
            input_dir = self.aligned_path / "equities" / "aligned_equities_1d" / f"ticker={ticker}"
            if not input_dir.exists():
                continue
                
            try:
                df = pd.read_parquet(input_dir)
                df = df.sort_values("date_utc")
                df.set_index("date_utc", inplace=True)
                
                # 5D Resample? 
                # '5d' frequency in pandas is 5 calendar days.
                # User wants "5D" patch (likely business days logic or simple window).
                # config says "patch_window": 5.
                # Creating non-overlapping 5-business-day windows is tricky with simple resample.
                # We'll use '5D' (calendar) or simple rolling? 
                # "patch_equities_5d" implies non-overlapping.
                # Let's use '5D' calendar for now or 'W'.
                # Given user constraint "patch_size": "5d".
                
                def first_s(s): return s.iloc[0] if len(s)>0 else np.nan
                def last_s(s): return s.iloc[-1] if len(s)>0 else np.nan
                
                hashed = df.resample("5D").agg({
                    "open": first_s,
                    "close": last_s,
                    "volume": "sum",
                    # keep index?
                })
                hashed["date_utc"] = hashed.index
                
                # RVOL (daily log returns)
                df["log_ret_1d"] = np.log(df["close"] / df["open"])
                df["sq_ret"] = df["log_ret_1d"] ** 2
                rvol_series = df["sq_ret"].resample("5D").sum().map(np.sqrt)
                
                patch_df = hashed.copy()
                patch_df["RVOL"] = rvol_series
                patch_df["RET"] = np.log(patch_df["close"] / patch_df["open"])
                patch_df.rename(columns={"volume": "VLM"}, inplace=True)
                
                # MISSING
                patch_df["FLOW"] = np.nan
                patch_df["SPREAD"] = np.nan
                patch_df["OI"] = np.nan
                patch_df["FUND"] = np.nan
                patch_df["BASIS"] = np.nan
                
                patch_df["ticker"] = ticker
                
                # Filter cols
                patch_df = patch_df[[
                    "date_utc", "ticker", "RET", "RVOL", "VLM", 
                    "FLOW", "SPREAD", "OI", "FUND", "BASIS"
                ]]
                
                save_partitioned(
                    patch_df,
                    "patch_equities_5d",
                    "patch",
                    base_path="data",
                    partition_cols=["ticker"]
                )
                
            except Exception as e:
                logger.error(f"Error building patch equities {ticker}: {e}")

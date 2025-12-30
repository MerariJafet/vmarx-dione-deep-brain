import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ingestion.stats")

class StatsGenerator:
    def __init__(self, config):
        self.config = config
        self.raw_path = Path(config["paths"]["raw"])
        self.stats_path = Path(config["paths"]["stats"])
        self.stats_path.mkdir(parents=True, exist_ok=True)

    def generate_crypto_stats(self, blocked_reasons=None):
        """
        Generates crypto_raw_coverage.json with:
        exchange, symbol, market_kind, min_ts, max_ts, rows, expected_bars, coverage_pct, blocked_reason
        """
        if blocked_reasons is None:
            blocked_reasons = {}

        crypto_path = self.raw_path / "crypto" / "raw_crypto_ohlcv_5m"
        
        # Requirements
        start_date = self.config["crypto_date_range"]["start"]
        end_date = self.config["crypto_date_range"]["end"]
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        # Expected bars calculation: (End - Start) / 5m + 1
        # 5m = 5 * 60 * 1000 ms = 300000 ms
        expected_bars = ((end_ts - start_ts) // 300000) + 1 
        # Actually proper candle count usually considers inclusive/exclusive. 
        # API usually returns candles starting at start_time.
        # If we request [start, end], how many 5m intervals fit?
        # ceil((end - start)/interval)? 
        # Prompt says: expected_bars = num_intervals_5m_between(start,end) + 1
        # Let's stick to that formula exactly.
        
        # Scan for what we have
        # Partition structure: exchange_code=... / symbol=... / year_month=... / part.parquet
        # It's better to load by exchange/symbol.
        
        exchanges = self.config["exchanges"]
        # Inverse map
        code_to_exch = {v: k for k, v in exchanges.items()}
        
        symbols = self.config["symbols"]["crypto"]
        
        stats = []
        
        # We need to iterate over expected combinations
        # We have BINANCE (Spot, Perp) and BYBIT (Spot, Linear) usually.
        # config.yaml doesn't explicitly list market_kinds per exchange, but downloader implies them.
        # We know: BINANCE->[spot, perp], BYBIT->[spot, linear]
        
        targets = [
            ("BINANCE", "spot"),
            ("BINANCE", "perp"),
            ("BYBIT", "spot"),
            ("BYBIT", "linear")
        ]
        
        for ex_name, kind in targets:
            ex_code = exchanges.get(ex_name)
            for symbol in symbols:
                # Check blocked reason
                blocked = blocked_reasons.get((ex_name, kind))
                
                # Check data
                # Path: raw/crypto/raw_crypto_ohlcv_5m/exchange_code={code}/symbol={symbol}
                # But wait, writer uses `partition_cols=['exchange_code', 'symbol', 'year_month']`
                # So paths are like `.../exchange_code=1/symbol=BTCUSDT/...`
                
                target_dir = crypto_path / f"exchange_code={ex_code}" / f"symbol={symbol}"
                
                rows = 0
                min_found = None
                max_found = None
                
                if target_dir.exists():
                    try:
                        # Read all parquet files in this dir (recursive year_month)
                        ds = pd.read_parquet(target_dir, columns=["timestamp_utc", "market_kind"])
                        # Filter by market_kind if needed?
                        # Partition doesn't include market_kind, so mixed spot/perp?
                        # Wait! Downloader saves to `raw_crypto_ohlcv_5m`.
                        # Structure: `exchange_code` / `symbol` / `year_month`.
                        # BUT market_kind is a column.
                        # Do we partition by market kind?
                        # The downloader `save_partitioned` call:
                        # `partition_cols=['exchange_code', 'symbol', 'year_month']`
                        # This means Spot and Perp for BTCUSDT on BINANCE are mixed in the same folder if they share exchange_code=1.
                        # This is a schema design issue potentially, OR we filter valid rows.
                        
                        # Let's check schema. If they are mixed, we must filter.
                        # Binance Spot exchange_code=1. Binance Perp exchange_code=1? 
                        # Usually they might be same code in this config? 
                        # checking downloader... `binance_ex_code` passed to both. Yes.
                        # So they are mixed.
                        
                        subset = ds[ds["market_kind"] == kind]
                        rows = len(subset)
                        if rows > 0:
                            min_found = subset["timestamp_utc"].min()
                            max_found = subset["timestamp_utc"].max()
                            if isinstance(min_found, pd.Timestamp):
                                min_found = int(min_found.timestamp() * 1000)
                            if isinstance(max_found, pd.Timestamp):
                                max_found = int(max_found.timestamp() * 1000)

                    except Exception as e:
                        logger.error(f"Error reading stats for {ex_name} {symbol}: {e}")
                
                # Calculate coverage
                coverage_pct = 0.0
                if expected_bars > 0:
                    coverage_pct = (rows / expected_bars) * 100.0
                
                stat_entry = {
                    "exchange_code": ex_code,
                    "symbol": symbol,
                    "market_kind": kind,
                    "min_ts": min_found,
                    "max_ts": max_found,
                    "rows": rows,
                    "expected_bars": expected_bars,
                    "coverage_pct": round(coverage_pct, 2),
                    "blocked_reason": blocked if blocked else None
                }
                stats.append(stat_entry)
        
        # Write JSON
        with open(self.stats_path / "crypto_raw_coverage.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Crypto stats generated at {self.stats_path / 'crypto_raw_coverage.json'}")

    def generate_equities_stats(self):
        """
        Generates equities_raw_coverage.json with:
        rows, min_date, max_date, missing_days estimate
        """
        equities_path = self.raw_path / "equities" / "raw_equities_1d"
        symbols = self.config["symbols"]["equities"]
        
        stats = []
        
        # Config range
        start_date_str = self.config["equities_date_range"]["start"]
        end_date_str = self.config["equities_date_range"]["end"]
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        
        # Business days calculation
        # pandas bdate_range
        expected_bdays = len(pd.bdate_range(start=start_dt, end=end_dt))
        
        for ticker in symbols:
            # Path: raw/equities/raw_equities_1d/ticker={ticker}
            target_dir = equities_path / f"ticker={ticker}"
            
            rows = 0
            min_date = None
            max_date = None
            
            if target_dir.exists():
                try:
                    ds = pd.read_parquet(target_dir, columns=["date_utc"])
                    rows = len(ds)
                    if rows > 0:
                        min_date = ds["date_utc"].min()
                        max_date = ds["date_utc"].max()
                        min_date = str(min_date).split(" ")[0]
                        max_date = str(max_date).split(" ")[0]
                except Exception as e:
                    logger.error(f"Error reading stats for {ticker}: {e}")
            
            # Missing estimate
            # This is rough, as stooq might have holidays etc.
            # But prompt asks for "missing trading days estimado".
            missing_est = max(0, expected_bdays - rows)
            
            stat_entry = {
                "ticker": ticker,
                "rows": rows,
                "min_date": min_date,
                "max_date": max_date,
                "missing_trading_days_est": missing_est
            }
            stats.append(stat_entry)
            
        with open(self.stats_path / "equities_raw_coverage.json", "w") as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Equities stats generated at {self.stats_path / 'equities_raw_coverage.json'}")

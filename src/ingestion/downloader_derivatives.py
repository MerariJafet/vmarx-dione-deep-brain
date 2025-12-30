import pandas as pd
from datetime import datetime
import logging
import time
from tqdm import tqdm
from src.ingestion.binance import BinanceClient
from src.ingestion.bybit import BybitClient
from src.storage.writer import save_partitioned

logger = logging.getLogger("ingestion.derivatives")

class DerivativesDownloader:
    def __init__(self, config):
        self.config = config
        self.binance = BinanceClient()
        self.bybit = BybitClient()
        # Read runtime status if injected
        self.status = config.get("_runtime_status", {
            "binance_spot": True, "binance_perp": True, "bybit": True, "metadata": {}
        })

    def run(self, start_date, end_date):
        symbols = self.config["symbols"]["crypto"]
        binance_ex = self.config["exchanges"]["BINANCE"]
        bybit_ex = self.config["exchanges"]["BYBIT"]
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        for symbol in tqdm(symbols, desc="Processing Derivatives"):
            if self.status["binance_perp"]:
                self._fetch_binance(symbol, binance_ex, start_ts, end_ts)
            
            if self.status["bybit"]:
                self._fetch_bybit(symbol, bybit_ex, start_ts, end_ts)

    def _fetch_binance(self, symbol, ex_code, start_ts, end_ts):
        # Funding Rate
        all_funding = []
        try:
            # Iteration logic tricky for funding if many, but usually limit=1000 covers a lot (8h per funding = 8000 hours ~ 1 year)
            # So one call might suffice for 6 months? 6 months * 3 funding/day = 540. Limit 1000 is enough.
            fund_data = self.binance.fetch_funding_rate(symbol, start_ts=start_ts, end_ts=end_ts)
            for f in fund_data:
                # symbol, fundingTime, fundingRate, markPrice
                all_funding.append({
                    "timestamp_utc": f['fundingTime'],
                    "exchange_code": ex_code,
                    "symbol": symbol,
                    "funding_rate": float(f['fundingRate']),
                    "open_interest": None, # Will merge later or separate row? 
                    # Schema has one row. If timestamps differ, we have sparse rows.
                    # Or we treat them as separate events aligned later?
                    # The schema suggests mixed columns. We can store sparse and forward fill later.
                    "liquidations_long": 0.0,
                    "liquidations_short": 0.0,
                    "ingested_at_utc": int(time.time()*1000),
                    "source": "BINANCE_API"
                })
        except Exception as e:
            logger.error(f"Binance funding error {symbol}: {e}")

        # Open Interest
        # /fapi/v1/openInterestHist period=5m
        # Limit 500 (max valid limit). 5m * 500 = 2500m = 41h. Need pagination.
        current_start = start_ts
        try:
            while current_start < end_ts:
                oi_data = self.binance.fetch_open_interest_hist(symbol, "5m", start_ts=current_start, limit=500)
                if not oi_data:
                    break
                    
                for o in oi_data:
                     # symbol, sumOpenInterest, sumOpenInterestValue, timestamp
                     all_funding.append({
                        "timestamp_utc": o['timestamp'],
                        "exchange_code": ex_code,
                        "symbol": symbol,
                        "funding_rate": None,
                        "open_interest": float(o['sumOpenInterest']), # or Value? usually size in coins is better for basis? Value is USDT. 
                        # sumOpenInterest is usually contracts/coins. 
                        # Let's save contracts count if possible, but Value is what Bybit often gives? 
                        # Actually Binance gives both. Let's use `sumOpenInterest` (quantity).
                        "liquidations_long": 0.0,
                        "liquidations_short": 0.0,
                        "ingested_at_utc": int(time.time()*1000),
                        "source": "BINANCE_API"
                    })
                
                last_ts = oi_data[-1]['timestamp']
                current_start = last_ts + 300000
                if current_start > end_ts:
                    break
        except Exception as e:
            logger.error(f"Binance OI error {symbol}: {e}")

        # Save
        if all_funding:
            self._save(all_funding, "raw_crypto_derivatives", "crypto")

    def _fetch_bybit(self, symbol, ex_code, start_ts, end_ts):
        all_rows = []
        # Funding History
        current_start = start_ts
        # Bybit funding limit 200. Frequency 8h usually. 200 * 8h = 1600h = 66 days. Need pagination.
        try:
            while current_start < end_ts:
                data = self.bybit.fetch_funding_history(symbol, start_ts=current_start, limit=200)
                if data.get("retCode") != 0: 
                    break
                rows = data.get("result", {}).get("list", [])
                if not rows:
                    break
                
                # Bybit returns NEWEST first? 
                rows = sorted(rows, key=lambda x: int(x['fundingRateTimestamp'])) 
                
                for r in rows:
                    all_rows.append({
                        "timestamp_utc": int(r['fundingRateTimestamp']),
                        "exchange_code": ex_code,
                        "symbol": symbol,
                        "funding_rate": float(r['fundingRate']),
                        "open_interest": None,
                        "liquidations_long": 0.0,
                        "liquidations_short": 0.0,
                        "ingested_at_utc": int(time.time()*1000),
                        "source": "BYBIT_API"
                    })
                
                last_ts = int(rows[-1]['fundingRateTimestamp'])
                current_start = last_ts + 1 # Next ms?
                if current_start >= end_ts:
                    break
        except Exception as e:
            logger.error(f"Bybit funding error {symbol}: {e}")

        # OI History
        # limit 200, interval 5m. 1000m = 16h. Pagination.
        current_start = start_ts
        try:
             while current_start < end_ts:
                data = self.bybit.fetch_open_interest(symbol, "5m", start_ts=current_start, limit=200)
                if data.get("retCode") != 0: break
                rows = data.get("result", {}).get("list", [])
                if not rows: break

                rows = sorted(rows, key=lambda x: int(x['timestamp']))

                for r in rows:
                     all_rows.append({
                        "timestamp_utc": int(r['timestamp']),
                        "exchange_code": ex_code,
                        "symbol": symbol,
                        "funding_rate": None,
                        "open_interest": float(r['openInterest']),
                        "liquidations_long": 0.0,
                        "liquidations_short": 0.0,
                        "ingested_at_utc": int(time.time()*1000),
                        "source": "BYBIT_API"
                    })
                
                last_ts = int(rows[-1]['timestamp'])
                # Bybit cursor can be tricky, using timestamp is safe if we sort
                current_start = last_ts + 300000 
                if current_start > end_ts: break

        except Exception as e:
            logger.error(f"Bybit OI error {symbol}: {e}")

        if all_rows:
            self._save(all_rows, "raw_crypto_derivatives", "crypto")

    def _save(self, data, table, domain):
        df = pd.DataFrame(data)
        # Drop strict duplicates
        df.drop_duplicates(subset=["timestamp_utc", "exchange_code", "symbol", "funding_rate", "open_interest"], inplace=True)
         
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
        df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], unit="ms", utc=True)
        save_partitioned(df, table, domain, self.base_path, partition_cols=['exchange_code', 'symbol'])

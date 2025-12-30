import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from tqdm import tqdm
from src.ingestion.binance import BinanceClient
from src.ingestion.bybit import BybitClient
from src.storage.writer import save_partitioned

logger = logging.getLogger("ingestion.ohlcv")

class OHLCVDownloader:
    def __init__(self, config):
        self.config = config
        self.binance = BinanceClient()
        self.bybit = BybitClient()
        self.base_path = config["paths"]["raw"]
        
        # Read runtime status if injected
        self.status = config.get("_runtime_status", {
            "binance_spot": True, "binance_perp": True, "bybit": True, "metadata": {}
        })
    
    def run(self, start_date, end_date):
        symbols = self.config["symbols"]["crypto"]
        binance_ex_code = self.config["exchanges"]["BINANCE"]
        bybit_ex_code = self.config["exchanges"]["BYBIT"]
        
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Preflight Checks
        blocked_reasons = {}
        
        # Binance Spot
        status_binance_spot = True
        if self.status["binance_spot"]:
            ok, reason = self.binance.check_connectivity("spot")
            if not ok:
                status_binance_spot = False
                blocked_reasons[("BINANCE", "spot")] = reason
                logger.warning(f"Skipping Binance Spot: {reason}")
        else:
            status_binance_spot = False

        # Binance Perp
        status_binance_perp = True
        if self.status["binance_perp"]:
            ok, reason = self.binance.check_connectivity("perp")
            if not ok:
                status_binance_perp = False
                blocked_reasons[("BINANCE", "perp")] = reason
                logger.warning(f"Skipping Binance Perp: {reason}")
        else:
            status_binance_perp = False

        # Bybit (Spot & Linear share same base usually, but good to check)
        status_bybit = True
        if self.status["bybit"]:
            # We assume BybitClient has a similar check? Or we just check a known endpoint.
            # We didn't update BybitClient yet, but let's assume valid property or check default
            # For now, let's implement a quick check here using ConnectivityChecker if BybitClient doesn't have it.
            # Or better, we check if we can reach bybit.
            from src.ingestion.connectivity import ConnectivityChecker
            # Bybit base defaults
            bybit_url = "https://api.bybit.com/v5/market/time"
            ok, reason = ConnectivityChecker.check_endpoint(bybit_url)
            if not ok:
                status_bybit = False
                blocked_reasons[("BYBIT", "spot")] = reason
                blocked_reasons[("BYBIT", "linear")] = reason
                logger.warning(f"Skipping Bybit: {reason}")
        else:
            status_bybit = False
            
        # Store blocked reasons for stats generator to use later? 
        # Ideally we pass this out or save it. 
        self.blocked_reasons = blocked_reasons

        for symbol in tqdm(symbols, desc="Processing Symbols"):
            # 1. Binance Spot
            if status_binance_spot:
                self._process_symbol(symbol, "5m", start_ts, end_ts, "BINANCE", binance_ex_code, "spot")
            
            # 2. Binance Perp
            if status_binance_perp:
                self._process_symbol(symbol, "5m", start_ts, end_ts, "BINANCE", binance_ex_code, "perp")
            
            # 3. Bybit Spot & Linear
            if status_bybit:
                self._process_symbol(symbol, "5m", start_ts, end_ts, "BYBIT", bybit_ex_code, "spot")
                self._process_symbol(symbol, "5m", start_ts, end_ts, "BYBIT", bybit_ex_code, "linear")
            
    def _process_symbol(self, symbol, interval, start_ts, end_ts, exchange_name, exchange_code, market_kind):
        logger.info(f"Downloading {exchange_name} {symbol} {market_kind}...")
        current_start = start_ts
        all_data = []
        
        while current_start < end_ts:
            try:
                if exchange_name == "BINANCE":
                    data = self.binance.fetch_klines(symbol, interval, start_ts=current_start, limit=1000, market_type=market_kind)
                    if not data: break
                    
                    for candle in data:
                        row = {
                            "timestamp_utc": candle[0],
                            "exchange_code": exchange_code,
                            "symbol": symbol,
                            "market_kind": market_kind,
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                            "quote_volume": float(candle[7]),
                            "trade_count": int(candle[8]),
                            "ingested_at_utc": int(time.time() * 1000),
                            "source": f"{exchange_name}_API"
                        }
                        all_data.append(row)
                    
                    last_open_time = data[-1][0]
                    current_start = last_open_time + 300000

                elif exchange_name == "BYBIT":
                    data_resp = self.bybit.fetch_kline(symbol, interval, start_ts=current_start, limit=200, category=market_kind)
                    if data_resp.get("retCode") != 0: break
                    candles = data_resp.get("result", {}).get("list", [])
                    if not candles: break
                    
                    candles = sorted(candles, key=lambda x: int(x[0]))
                    for candle in candles:
                        row = {
                            "timestamp_utc": int(candle[0]),
                            "exchange_code": exchange_code,
                            "symbol": symbol,
                            "market_kind": market_kind,
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                            "quote_volume": float(candle[6]),
                            "trade_count": 0,
                            "ingested_at_utc": int(time.time() * 1000),
                            "source": f"{exchange_name}_API"
                        }
                        all_data.append(row)
                        
                    last_ts = int(candles[-1][0])
                    current_start = last_ts + 300000
                
            except Exception as e:
                logger.error(f"Error fetching {symbol} {market_kind}: {e}")
                # If error is connectivity related, maybe we should stop trying for this domain?
                # For now, break loop for this symbol
                break
                
            if current_start >= end_ts: break
                
        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=["timestamp_utc", "exchange_code", "symbol", "market_kind"], inplace=True)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True)
            df["ingested_at_utc"] = pd.to_datetime(df["ingested_at_utc"], unit="ms", utc=True)
            # PARTITION FIX
            save_partitioned(df, "raw_crypto_ohlcv_5m", "crypto", self.base_path, partition_cols=['exchange_code', 'symbol', 'year_month'])

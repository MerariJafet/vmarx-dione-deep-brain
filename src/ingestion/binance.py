import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datetime import datetime
import logging
import time

logger = logging.getLogger("ingestion.binance")

class BinanceClient:
    def __init__(self, use_auth=False, api_key=None, api_secret=None):
        self.spot_base_urls = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com", "https://api2.binance.com"]
        self.futures_base_urls = ["https://fapi.binance.com", "https://fapi.binance.vision"]
        self.spot_base_url = self.spot_base_urls[0]
        self.futures_base_url = self.futures_base_urls[0]
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        })
        self.session.verify = False # Bypass SSL if certs issue
        requests.packages.urllib3.disable_warnings() # Silence warnings

    def check_connectivity(self, market_type="spot"):
        from src.ingestion.connectivity import ConnectivityChecker
        
        urls_to_try = self.spot_base_urls if market_type == "spot" else self.futures_base_urls
        path = "/api/v3/ping" if market_type == "spot" else "/fapi/v1/ping"
        
        for base_url in urls_to_try:
            check_url = f"{base_url}{path}"
            is_ok, reason = ConnectivityChecker.check_endpoint(check_url, timeout=3)
            if is_ok:
                # Update current active base URL
                if market_type == "spot":
                    self.spot_base_url = base_url
                    logger.info(f"Binance Spot connected via {base_url}")
                else:
                    self.futures_base_url = base_url
                    logger.info(f"Binance Futures connected via {base_url}")
                return True, "OK"
            else:
                logger.warning(f"Failed to connect to {base_url}: {reason}")
        
        # If all fail, return the last reason (likely DNS_BLOCKED or HTTP_403)
        return False, reason

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, Exception))
    )
    def _get(self, url, params=None):
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit hit (429). Backing off.")
                time.sleep(5) # Manual extended sleep
                raise e
            logger.error(f"HTTP Error {e.response.status_code} for {url}")
            raise e

    def fetch_klines(self, symbol, interval, start_ts=None, end_ts=None, limit=1000, market_type="spot"):
        """
        Fetch OHLCV candles. 
        Timestamp in ms.
        market_type: 'spot' or 'perp' (futures)
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)

        if market_type == "spot":
            url = f"{self.spot_base_url}/api/v3/klines"
        else:
            url = f"{self.futures_base_url}/fapi/v1/klines"

        return self._get(url, params)

    def fetch_funding_rate(self, symbol, start_ts=None, end_ts=None, limit=1000):
        # /fapi/v1/fundingRate
        params = {"symbol": symbol, "limit": limit}
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)
        
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        return self._get(url, params)

    def fetch_open_interest_hist(self, symbol, period, limit=30, start_ts=None, end_ts=None):
        # /fapi/v1/openInterestHist
        # period e.g. "5m"
        params = {"symbol": symbol, "period": period, "limit": limit}
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)
            
        url = f"{self.futures_base_url}/fapi/v1/openInterestHist"
        return self._get(url, params)
    
    def fetch_book_ticker(self, symbol, market_type="spot"):
        params = {"symbol": symbol}
        if market_type == "spot":
            url = f"{self.spot_base_url}/api/v3/ticker/bookTicker"
        else:
            url = f"{self.futures_base_url}/fapi/v1/ticker/bookTicker"
        return self._get(url, params)
    
    def fetch_agg_trades(self, symbol, start_ts=None, end_ts=None, limit=1000, market_type="spot"):
        params = {"symbol": symbol, "limit": limit}
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)
            
        if market_type == "spot":
            url = f"{self.spot_base_url}/api/v3/aggTrades"
        else:
            url = f"{self.futures_base_url}/fapi/v1/aggTrades"
        return self._get(url, params)

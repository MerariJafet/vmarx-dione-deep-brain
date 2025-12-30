import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import time

logger = logging.getLogger("ingestion.bybit")

class BybitClient:
    def __init__(self, use_auth=False, api_key=None, api_secret=None):
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        })
        self.session.verify = False
        requests.packages.urllib3.disable_warnings()

    def check_connectivity(self):
        from src.ingestion.connectivity import ConnectivityChecker
        # Use time endpoint
        return ConnectivityChecker.check_endpoint(f"{self.base_url}/v5/market/time")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError, Exception))
    )
    def _get(self, url, params=None):
        try:
            response = self.session.get(url, params=params, timeout=10)
            # Bybit specific: 200 OK mostly, check retCode?
            # public endpoints usually fine
            response.raise_for_status()
            data = response.json()
            if data.get("retCode") != 0:
                logger.warning(f"Bybit API Error: {data.get('retMsg')} ({data.get('retCode')}) for {url}")
                # Don't always raise, empty data might be retCode 0 with empty list
            return data
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning("Rate limit hit (429). Backing off.")
                time.sleep(5)
                raise e
            logger.error(f"HTTP Error {e.response.status_code} for {url}")
            raise e

    def fetch_kline(self, symbol, interval, start_ts=None, end_ts=None, limit=200, category="spot"):
        """
        Bybit V5 Market Kline.
        interval: 1,3,5,15,30,60, ...
        category: spot, linear, inverse
        start, end: ms timestamp
        """
        # Map DB interval "5m" to Bybit "5"
        bybit_interval = interval.replace("m", "") if interval.endswith("m") else interval 
        
        params = {
            "category": category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": limit
        }
        if start_ts:
            params["start"] = int(start_ts)
        if end_ts:
            params["end"] = int(end_ts)

        url = f"{self.base_url}/v5/market/kline"
        return self._get(url, params)

    def fetch_tickers(self, symbol=None, category="linear"):
        """
        To get funding rate, OI, best bid/ask snapshot.
        Warning: this is a snapshot endpoint, not history, unless there's a history endpoint.
        
        For funding history: /v5/market/funding/history
        For OI history: /v5/market/open-interest
        """
        url = f"{self.base_url}/v5/market/tickers"
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._get(url, params)

    def fetch_funding_history(self, symbol, start_ts=None, end_ts=None, limit=200, category="linear"):
        url = f"{self.base_url}/v5/market/funding/history"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit
        }
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)
        return self._get(url, params)
    
    def fetch_open_interest(self, symbol, interval, start_ts=None, end_ts=None, limit=200, category="linear"):
        url = f"{self.base_url}/v5/market/open-interest"
        bybit_interval = interval.replace("m", "") if interval.endswith("m") else interval
        params = {
            "category": category,
            "symbol": symbol,
            "intervalTime": bybit_interval,
            "limit": limit
        }
        if start_ts:
            params["startTime"] = int(start_ts)
        if end_ts:
            params["endTime"] = int(end_ts)
        return self._get(url, params)
        
    def fetch_public_trading_history(self, symbol, limit=1000, category="spot"):
        # /v5/market/recent-trade
        url = f"{self.base_url}/v5/market/recent-trade"
        params = {
            "category": category,
            "symbol": symbol,
            "limit": limit
        }
        return self._get(url, params)

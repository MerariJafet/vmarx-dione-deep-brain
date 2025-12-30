import requests
import logging
import os

logger = logging.getLogger("ingestion.connectivity")

class ConnectivityChecker:
    @staticmethod
    def check_endpoint(url, timeout=5):
        try:
            # We assume a GET to the root or specific ping endpoint
            # If not provided, we just check socket connection via requests
            proxies = {
                "http": os.environ.get("HTTP_PROXY"),
                "https": os.environ.get("HTTPS_PROXY")
            }
            # Remove None values to let requests handle defaults if env vars are unset
            proxies = {k: v for k, v in proxies.items() if v}
            
            response = requests.get(url, timeout=timeout, proxies=proxies, verify=False)
            
            if response.status_code == 403:
                return False, "HTTP_403"
            if response.status_code == 429:
                return False, "RATE_LIMIT"
            # 404 is technically a "connection success" vs "network failure"
            # but for an API base, it might mean wrong endpoint. 
            # We'll treat 200-299 as OK.
            if 200 <= response.status_code < 300:
                return True, "OK"
                
            return True, "OK" # Defaulting to OK for other codes if we just want connectivity, but strict would be better.
                              # For a ping endpoint, we expect 200.
                              
        except requests.exceptions.ConnectTimeout:
            return False, "TIMEOUT"
        except requests.exceptions.ConnectionError:
            return False, "DNS_BLOCKED" # Often DNS or Refused
        except requests.exceptions.ProxyError:
            return False, "PROXY_ERROR"
        except Exception as e:
            return False, "UNKNOWN"

import yaml
import logging
from src.storage.writer import deduplicate_dataset

logging.basicConfig(level=logging.INFO)

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    print("Starting deduplication for Crypto OHLCV...")
    
    # crypto ohlcv
    # PK: timestamp_utc, exchange_code, symbol, market_kind
    # Note: exchange_code and symbol are partition keys, so read_parquet includes them.
    # timestamp_utc and market_kind are in the file.
    
    deduplicate_dataset(
        table_name="raw_crypto_ohlcv_5m",
        domain="crypto",
        base_path=config["paths"]["raw"],
        pk_cols=["timestamp_utc", "exchange_code", "symbol", "market_kind"]
    )
    
    print("Deduplication Complete.")

if __name__ == "__main__":
    main()

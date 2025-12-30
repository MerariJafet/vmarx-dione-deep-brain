import argparse
import sys
import yaml
from pathlib import Path
import logging

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.ingestion.downloader_ohlcv import OHLCVDownloader
from src.ingestion.downloader_derivatives import DerivativesDownloader
from src.ingestion.stooq import StooqDownloader
from src.utils.stats import generate_crypto_stats, generate_equities_stats

logger = get_logger("runner")

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_all(args, config):
    logger.info("Starting run_all job...")
    run_domain_crypto(args, config)
    run_domain_equities(args, config)
    logger.info("Job run_all completed.")

def run_domain_crypto(args, config):
    logger.info("Starting run_domain_crypto...")
    
    start_date = config["crypto_date_range"]["start"]
    end_date = config["crypto_date_range"]["end"]
    
    # Metadata for stats
    # Keys: (exchange_code, symbol, market_kind) -> reason
    # Broad keys: (exchange_code, None, None) -> reason
    run_metadata = {} 
    
    # Preflight Checks
    # 1. Binance
    from src.ingestion.binance import BinanceClient
    bn = BinanceClient()
    ok_spot, reason_spot = bn.check_connectivity("spot")
    if not ok_spot:
        logger.error(f"Binance Spot Blocked: {reason_spot}")
        run_metadata[(1, None, None)] = reason_spot # Broad block? Or just spot.
        # Mark all spot symbols blocked?
        # Better: Downloaders should check this metadata or pass it in.
    
    ok_fut, reason_fut = bn.check_connectivity("futures")
    if not ok_fut:
        logger.error(f"Binance Futures Blocked: {reason_fut}")
        run_metadata[(1, None, "perp")] = reason_fut

    # 2. Bybit
    from src.ingestion.bybit import BybitClient
    by = BybitClient()
    ok_by, reason_by = by.check_connectivity()
    if not ok_by:
         logger.error(f"Bybit Blocked: {reason_by}")
         run_metadata[(2, None, None)] = reason_by
    
    # Pass metadata/flags to downloaders? 
    # Currently downloaders instantiate their own clients. 
    # We can inject the status into config or pass as args.
    # Hack: Inject into config temporarily for this run object
    config["_runtime_status"] = {
        "binance_spot": ok_spot,
        "binance_perp": ok_fut,
        "bybit": ok_by,
        "metadata": run_metadata
    }

    # Run OHLCV
    logger.info(f"Starting OHLCV Download...")
    ohlcv = OHLCVDownloader(config)
    ohlcv.run(start_date, end_date)
    
    # Run Derivatives
    # Only if futures is OK or Bybit matches
    if ok_fut or ok_by:
        logger.info(f"Starting Derivatives Download...")
        deriv = DerivativesDownloader(config)
        deriv.run(start_date, end_date)
    else:
        logger.warning("Skipping Derivatives: All sources blocked.")

    # Generate Stats
    generate_crypto_stats(config, run_metadata)
    
    logger.info("Crypto pipeline completed.")

def run_domain_equities(args, config):
    logger.info("Starting run_domain_equities...")
    
    stooq = StooqDownloader(config)
    stooq.run()
    
    generate_equities_stats(config)
    logger.info("Equities pipeline completed.")

def backfill_range(args, config):
    pass # reusing component logic as needed

def main():
    parser = argparse.ArgumentParser(description="VMarx Dione DB Job Runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_run_all = subparsers.add_parser("run_all", help="Run full pipeline")
    parser_run_all.set_defaults(func=run_all)

    parser_crypto = subparsers.add_parser("run_domain_crypto", help="Crypto")
    parser_crypto.set_defaults(func=run_domain_crypto)

    parser_equities = subparsers.add_parser("run_domain_equities", help="Equities")
    parser_equities.set_defaults(func=run_domain_equities)

    args = parser.parse_args()
    
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("Config file not found")
        sys.exit(1)

    if hasattr(args, 'func'):
        args.func(args, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

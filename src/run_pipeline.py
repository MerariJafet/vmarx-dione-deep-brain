import yaml
import logging
import sys
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("runner")

from src.ingestion.downloader_ohlcv import OHLCVDownloader
from src.ingestion.stooq import StooqDownloader
from src.ingestion.stats import StatsGenerator

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

from src.processing.aligned import AlignedLayerBuilder
from src.processing.patch import PatchLayerBuilder
from src.processing.events import EventGenerator
from src.storage.writer import deduplicate_dataset
import shutil

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VMarx Pipeline Runner")
    parser.add_argument("--stage", type=str, default="all", help="Comma-separated stages: ingest, aligned, patch, events, stats, export, tokenize. 'all' runs everything.")
    parser.add_argument("--domain", type=str, choices=["crypto", "equities", "all"], default="all", help="Domain to process: crypto, equities, or all.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    args = parser.parse_args()

    config = load_config()

    stages = args.stage.split(",")
    if "all" in stages:
        stages = ["ingest", "aligned", "patch", "events", "stats", "export", "tokenize"]
    
    selected_domain = args.domain

    def should_run(stage_name):
        return stage_name in stages

    if args.dry_run:
        logger.info(f"DRY RUN MODE. Stages selected: {stages}. Domain: {selected_domain}")

    # 1. Crypto Ingestion (Sprint 2.1)
    if should_run("ingest") and selected_domain in ["crypto", "all"]:
        logger.info("=== [Stage: Ingest] Crypto Ingestion ===")
        if not args.dry_run:
            start_crypto = config["crypto_date_range"]["start"]
            end_crypto = config["crypto_date_range"]["end"]
            # crypto_downloader = OHLCVDownloader(config)
            # crypto_downloader.run(start_crypto, end_crypto)

    # 1.5. Data Integrity
    # if should_run("ingest") and selected_domain in ["crypto", "all"]:
        # logger.info("=== Integrity Check (Dedupe) ===")
        # if not args.dry_run:
             # deduplicate_dataset("raw_crypto_ohlcv_5m", "crypto", config["paths"]["raw"])
    
    # 2. Equities Ingestion
    if should_run("ingest") and selected_domain in ["equities", "all"]:
        logger.info("=== [Stage: Ingest] Equities Ingestion ===")
        # if not args.dry_run:
            # stooq_downloader = StooqDownloader(config)
            # stooq_downloader.run()

    # 3. Aligned Layer
    if should_run("aligned"):
        logger.info("=== [Stage: Aligned] Building Aligned Layer ===")
        if not args.dry_run:
            aligned_builder = AlignedLayerBuilder(config)
            if selected_domain in ["crypto", "all"]:
                aligned_builder.build_crypto_5m()
            if selected_domain in ["equities", "all"]:
                aligned_builder.build_equities_1d()
    
    # 4. Patch Layer
    if should_run("patch"):
        logger.info("=== [Stage: Patch] Building Patch Layer ===")
        if not args.dry_run:
            patch_builder = PatchLayerBuilder(config)
            if selected_domain in ["crypto", "all"]:
                patch_builder.build_crypto_1h()
            if selected_domain in ["equities", "all"]:
                patch_builder.build_equities_5d()
    
    # 5. Events
    if should_run("events"):
        logger.info("=== [Stage: Events] Generating Events ===")
        if not args.dry_run:
            event_gen = EventGenerator(config)
            # Event generator currently runs all; assuming internal optimization or check config
            event_gen.generate_events() 

    # 6. Stats Generation
    if should_run("stats"):
        logger.info("=== [Stage: Stats] Generating Stats ===")
        if not args.dry_run:
            stats_gen = StatsGenerator(config)
            if selected_domain in ["crypto", "all"]:
                stats_gen.generate_crypto_stats(blocked_reasons={})
            if selected_domain in ["equities", "all"]:
                stats_gen.generate_equities_stats()
    
    # 7. Export Pilot
    if should_run("export"):
        logger.info("=== [Stage: Export] Exporting Pilot Dataset ===")
        if not args.dry_run:
            export_dir = Path("data/dataset_pilot_v0_1")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                def copy_dir(src_name, dest_name):
                    s = Path(f"data/patch/{src_name}")
                    if not s.exists(): 
                        s = Path(f"data/events/{src_name}")
                    d = export_dir / dest_name
                    if s.exists():
                        if d.exists(): shutil.rmtree(d)
                        shutil.copytree(s, d)
                        
                if selected_domain in ["crypto", "all"]:
                    copy_dir("patch_crypto_1h", "patch_crypto_1h")
                    copy_dir("events_crypto", "events_crypto")
                if selected_domain in ["equities", "all"]:
                    copy_dir("patch_equities_5d", "patch_equities_5d")
                    copy_dir("events_equities", "events_equities")
                
                # Copy stats
                if should_run("stats"): # Only copy if stats ran or exist? Just copy whatever is there
                    stats_dest = export_dir / "stats"
                    stats_dest.mkdir(exist_ok=True)
                    for f in Path("data/stats").glob("*.json"):
                        shutil.copy(f, stats_dest / f.name)
            except Exception as e:
                logger.error(f"Export failed: {e}")
    
    # 8. Tokenization
    if should_run("tokenize"):
        logger.info("=== [Stage: Tokenize] Starting Tokenization ===")
        
        from src.tokenization.dictionary import TokenDictionary
        from src.tokenization.registry import AssetRegistry
        from src.tokenization.binner import FeatureBinner
        from src.tokenization.serializer import SequenceSerializer
        import numpy as np
        import pandas as pd
        
        if not args.dry_run:
            # 8.1 Build Registry (Needs all patches?)
            # Registry should be additive or robust? 
            # Rebuilding registry only for selected domain might break ID consistency if it's not careful.
            # Assuming registry scans disk paths.
            logger.info("Building Asset Registry...")
            registry = AssetRegistry()
            patch_paths = []
            if selected_domain in ["crypto", "all"]:
                patch_paths.append("data/patch/patch_crypto_1h")
            if selected_domain in ["equities", "all"]:
                patch_paths.append("data/patch/patch_equities_5d")
            
            registry.build_registry(patch_paths)
            
            # 8.2 Fit Binner (Ideally fit on all if possible, or load existing?)
            # Binner fits quantiles. If we just fit on equities, might shift distribution?
            # Assuming we want to use existing fit or fit on available data.
            logger.info("Fitting Feature Binner...")
            binner = FeatureBinner()
            fit_data = {}
            
            try:
                # Always allow loading crypto data for consistent binning if present
                if Path("data/patch/patch_crypto_1h").exists():
                     # Only load if we are processing all OR we want robust stats?
                     # Let's just load what's available to ensure Binner is robust.
                     crypto_df = pd.read_parquet("data/patch/patch_crypto_1h")
                     fit_data["crypto"] = crypto_df
            except Exception: pass
            
            try:
                if Path("data/patch/patch_equities_5d").exists():
                    eq_df = pd.read_parquet("data/patch/patch_equities_5d")
                    fit_data["equities"] = eq_df
            except Exception: pass
            
            if fit_data:
                binner.fit(fit_data)
            else:
                logger.warning("No data to fit binner!")

            # 8.3 Serialize
            logger.info("Running Tokenization Serializer...")
            events_crypto = pd.read_parquet("data/events/events_crypto") if Path("data/events/events_crypto").exists() else pd.DataFrame()
            events_equities = pd.read_parquet("data/events/events_equities") if Path("data/events/events_equities").exists() else pd.DataFrame()
            
            dictionary = TokenDictionary()
            serializer = SequenceSerializer(dictionary, binner)
            
            output_dir = Path("data/tokenized")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process Crypto
            if selected_domain in ["crypto", "all"] and "crypto" in fit_data:
                df = fit_data["crypto"]
                if "symbol" in df.columns:
                    for symbol in df["symbol"].unique():
                        # Optimize: Skip if exists? No, user asked for tokenize command.
                        asset_df = df[df["symbol"] == symbol]
                        if not events_crypto.empty and "symbol" in events_crypto.columns:
                            asset_events = events_crypto[events_crypto["symbol"] == symbol]
                        else:
                            asset_events = pd.DataFrame()
                        
                        try:
                            a_id = registry.get_id(symbol)
                            sequences = serializer.tokenize_asset(
                                a_id, 
                                "CRYPTO_SPOT_5M_PATCH_1H", 
                                "CRYPTO_SPOT_OHLCV_5M", 
                                asset_df, 
                                asset_events
                            )
                            for i, seq in enumerate(sequences):
                                np.save(output_dir / f"crypto_{symbol}_{i}.npy", seq)
                        except Exception as e:
                            logger.error(f"Error tokenizing {symbol}: {e}")
                        
            # Process Equities
            if selected_domain in ["equities", "all"] and "equities" in fit_data:
                df = fit_data["equities"]
                # FIX: Rename date_utc to timestamp_utc if needed
                if "date_utc" in df.columns and "timestamp_utc" not in df.columns:
                    logger.info("Renaming date_utc -> timestamp_utc for equities")
                    df = df.rename(columns={"date_utc": "timestamp_utc"})

                if "ticker" in df.columns:
                    for ticker in df["ticker"].unique():
                        asset_df = df[df["ticker"] == ticker]
                        if not events_equities.empty and "ticker" in events_equities.columns:
                            asset_events = events_equities[events_equities["ticker"] == ticker]
                        else:
                            asset_events = pd.DataFrame()
                        
                        try:
                            a_id = registry.get_id(ticker)
                            sequences = serializer.tokenize_asset(
                                a_id, 
                                "EQUITIES_1D_PATCH_5D", 
                                "EQUITIES_DAILY_1D", 
                                asset_df, 
                                asset_events
                            )
                            for i, seq in enumerate(sequences):
                                np.save(output_dir / f"equities_{ticker}_{i}.npy", seq)
                        except Exception as e:
                            logger.error(f"Error tokenizing {ticker}: {e}")

    logger.info("=== Pipeline Complete ===")
    
    # Summary Print
    if not args.dry_run:
        print("\n\n=== PIPELINE SUMMARY ===")
        print(f"Export location: data/dataset_pilot_v0_1")

if __name__ == "__main__":
    main()

import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("tokenization.registry")

class AssetRegistry:
    def __init__(self, output_dir="data/tokenization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_map = {}
        
    def build_registry(self, patch_dirs):
        """
        Scans patch directories to find all unique symbols/tickers.
        patch_dirs: list of Paths to scan.
        """
        symbols = set()
        
        for p_dir in patch_dirs:
            p_path = Path(p_dir)
            if not p_path.exists():
                logger.warning(f"Path {p_path} does not exist.")
                continue
            
            # Read all parquet files to find symbols?
            # Or just assume filenames or partition structure?
            # Pilot structure: data/patch/patch_crypto_1h/ (partitioned?) 
            # or single file?
            # Writer uses save_partitioned, but for patch layer, we used simple save_partitioned
            # which might have saved as single file if no partition cols, 
            # OR as dataset.
            # In Sprint 4/5 we saved patch layers.
            # Let's check if they are directories with parquet files.
            # Usually we read the dataset and get unique symbols.
            
            try:
                # If it's a directory of parquets
                ds = pd.read_parquet(p_path)
                # Check column "symbol" or "ticker"
                if "symbol" in ds.columns:
                    symbols.update(ds["symbol"].unique())
                elif "ticker" in ds.columns:
                    symbols.update(ds["ticker"].unique())
            except Exception as e:
                logger.error(f"Error scanning {p_path}: {e}")
                
        # Sort stable
        sorted_symbols = sorted(list(symbols))
        
        # Assign IDs
        self.asset_map = {sym: i for i, sym in enumerate(sorted_symbols)}
        
        # Save
        with open(self.output_dir / "asset_map.json", "w") as f:
            json.dump(self.asset_map, f, indent=2)
            
        logger.info(f"Asset Registry built. {len(self.asset_map)} assets found.")
        
    def load(self):
        try:
            with open(self.output_dir / "asset_map.json", "r") as f:
                self.asset_map = json.load(f)
        except FileNotFoundError:
            logger.warning("Asset map not found. Build it first.")
            
    def get_id(self, symbol):
        return self.asset_map.get(symbol)

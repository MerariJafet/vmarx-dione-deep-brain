import logging
import sys
from pathlib import Path
from src.training.dataset import SlidingWindowDataset

logger = logging.getLogger("training.preflight")

def check_dataset_health(data_dir, train_pct=0.85):
    """
    Asserts that:
    1. Total Equity files > 0
    2. Val Equity files > 0 (using actual split logic)
    """
    logger.info("=== Starting Dataset Preflight Check ===")
    
    # Use the actual dataset logic to simulate split
    # We create a dummy instance just to use its discovery logic?
    # Or just use the logic directly. 
    # Let's instantiate the dataset in 'val' mode without iteration.
    
    try:
        # Check Total counts first
        p = Path(data_dir)
        all_files = list(p.glob("*.npy"))
        crypto_files = [f for f in all_files if f.name.startswith("crypto_")]
        equity_files = [f for f in all_files if f.name.startswith("equities_")] # or "equity_" check both?
        # User confirmed "equities_" prefix in previous turn.
        
        total_crypto = len(crypto_files)
        total_equity = len(equity_files)
        
        logger.info(f"Total Files Found: {len(all_files)}")
        logger.info(f"  - Crypto: {total_crypto}")
        logger.info(f"  - Equity: {total_equity}")
        
        if total_equity == 0:
            logger.error("CRITICAL: No Equities data found (total_equity=0).")
            return False

        # Check Validation Split
        # We need to simulate the split logic exactly as run during training.
        # SlidingWindowDataset uses _discover_and_split_files
        
        val_ds = SlidingWindowDataset(data_dir, split="val", train_pct=train_pct)
        # We can inspect val_ds.files
        
        val_files = val_ds.files
        val_crypto = [f for f in val_files if f.name.startswith("crypto_")]
        val_equity = [f for f in val_files if f.name.startswith("equities_")]
        
        logger.info(f"Validation Split Files: {len(val_files)}")
        logger.info(f"  - Val Crypto: {len(val_crypto)}")
        logger.info(f"  - Val Equity: {len(val_equity)}")
        
        if len(val_equity) == 0:
            logger.error("CRITICAL: Validation Equity count is 0! Split logic excluded all equity files.")
            return False
            
        logger.info("=== Dataset Preflight Check PASSED ===")
        return True

    except Exception as e:
        logger.error(f"Preflight Check Failed with Exception: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not check_dataset_health("data/tokenized"):
        sys.exit(1)

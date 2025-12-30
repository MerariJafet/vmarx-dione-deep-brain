import logging
import os
import json
import yaml
import torch.utils.checkpoint

# MONKEYPATCH: Enforce use_reentrant=False to silence PyTorch warnings
if hasattr(torch.utils.checkpoint, "checkpoint"):
    _original_checkpoint = torch.utils.checkpoint.checkpoint
    def checkpoint_patched(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return _original_checkpoint(function, *args, **kwargs)
    torch.utils.checkpoint.checkpoint = checkpoint_patched

from pathlib import Path
from src.training.dataset import SlidingWindowDataset
from src.training.model_utils import get_qlora_model
from src.training.trainer import PilotTrainer
from src.training.preflight import check_dataset_health
import argparse
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training.runner")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Set rigid seed for recovery (ensure different from previous run)
    RECOVERY_SEED = 97
    import random
    import numpy as np
    import torch
    random.seed(RECOVERY_SEED)
    np.random.seed(RECOVERY_SEED)
    torch.manual_seed(RECOVERY_SEED)
    torch.cuda.manual_seed_all(RECOVERY_SEED)
    logger.info(f"*** RECOVERY MODE *** Set Global Seed to {RECOVERY_SEED}")

    # Load spec
    with open("configs/training_resume_test.json", "r") as f:
        full_config = json.load(f)
    # Select Config V1.0 if available
    if "training_v1_0" in full_config:
        config = full_config["training_v1_0"]
        logger.info("Using configuration: training_v1_0")
    else:
        config = full_config["pilot_training_v0.1"]
        logger.warning("V1.0 config not found, falling back to Pilot.")
    
    # 1. Datasets
    logger.info("Initializing Datasets...")
    data_dir = config["paths"]["tokenized_data"]
    sampling_mode = config["data"]["sampling"]["crypto_equity_balance"]
    
    # Train
    train_dataset = SlidingWindowDataset(
        data_dir, 
        seq_len=config["training_setup"]["seq_len_effective"],
        stride=config["training_setup"].get("stride", 1024),
        split="train",
        train_pct=config["data"]["split"]["train_pct"],
        sampling_mode=sampling_mode
    )
    
    # Val - Crypto
    val_crypto = SlidingWindowDataset(
        data_dir, 
        seq_len=config["training_setup"]["seq_len_effective"],
        split="val",
        train_pct=config["data"]["split"]["train_pct"],
        domain_filter="crypto"
    )

    # Val - Equity
    val_equity = SlidingWindowDataset(
        data_dir, 
        seq_len=config["training_setup"]["seq_len_effective"],
        split="val",
        train_pct=config["data"]["split"]["train_pct"],
        domain_filter="equities" # Assumes files start with equities_
    )
    
    # DataLoaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training_setup"]["microbatch"],
        num_workers=2,
        pin_memory=True
    )
    
    val_loaders = {}
    if len(val_crypto.files) > 0:
        val_loaders["crypto"] = DataLoader(val_crypto, batch_size=config["training_setup"]["microbatch"], num_workers=1)
    
    if len(val_equity.files) > 0:
        val_loaders["equity"] = DataLoader(val_equity, batch_size=config["training_setup"]["microbatch"], num_workers=1)
    
    if not val_loaders:
        logger.warning("No validation data found!")

    # 2. Model
    logger.info("Initializing Model (QLoRA)...")
    model = get_qlora_model(config)
    
    # 3. Check for Resume
    output_dir = config["paths"]["output_dir"]
    resume_path = None
    if os.path.exists(output_dir):
        # Look for checkpoints
        # Filter for directories like 'checkpoint_123' where the part after underscore is an integer
        ckpts = [d for d in os.listdir(output_dir) 
                 if d.startswith("checkpoint_") 
                 and len(d.split("_")) > 1 
                 and d.split("_")[1].isdigit()]
        
        if ckpts:
            # Sort by step number
            ckpts.sort(key=lambda x: int(x.split("_")[1]))
            resume_path = os.path.join(output_dir, ckpts[-1])
            logger.info(f"Found existing checkpoints. Will RESUME from: {resume_path}")

    # 4. Trainer
    logger.info("Starting Trainer...")
    trainer = PilotTrainer(model, train_loader, config, output_dir, val_loaders=val_loaders)
    trainer.train(resume_from_checkpoint=resume_path)

if __name__ == "__main__":
    main()

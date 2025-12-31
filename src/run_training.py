import logging
import os
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.training.model_utils import get_qlora_model, get_tokenizer
from src.training.trainer import PilotTrainer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("training.runner")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_spec.json")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        full_config = json.load(f)
    
    # Select first key or use specific one
    config_key = list(full_config.keys())[0] if "training_v1_0" not in full_config else "training_v1_0"
    if "training_v2_0" in full_config: config_key = "training_v2_0"
    
    config = full_config[config_key]
    logger.info(f"Using configuration: {config_key}")

    # Initialize Tokenizer
    tokenizer = get_tokenizer(config)

    # 1. Dataset Selection
    if "instruction_data" in config["paths"]:
        logger.info("Initializing Instruction Dataset (SFT)...")
        from src.training.instruction_dataset import InstructionDataset, instruction_collate_fn
        train_path = os.path.join(config["paths"]["instruction_data"], "train.jsonl")
        val_path = os.path.join(config["paths"]["instruction_data"], "val.jsonl")
        
        train_dataset = InstructionDataset(train_path, tokenizer, max_seq_length=config["training_setup"]["seq_len_effective"])
        val_dataset = InstructionDataset(val_path, tokenizer, max_seq_length=config["training_setup"]["seq_len_effective"])
        
        collate_fn = instruction_collate_fn
        val_loaders = {"instruction": DataLoader(val_dataset, batch_size=config["training_setup"]["microbatch"], collate_fn=collate_fn)}
    else:
        logger.info("Initializing Sliding Window Dataset (Stage 1)...")
        from src.training.dataset import SlidingWindowDataset, custom_collate_fn
        data_dir = config["paths"]["tokenized_data"]
        train_dataset = SlidingWindowDataset(data_dir, seq_len=config["training_setup"]["seq_len_effective"], split="train")
        val_crypto = SlidingWindowDataset(data_dir, seq_len=config["training_setup"]["seq_len_effective"], split="val", domain_filter="crypto")
        
        collate_fn = custom_collate_fn
        val_loaders = {"crypto": DataLoader(val_crypto, batch_size=config["training_setup"]["microbatch"], collate_fn=collate_fn)}

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training_setup"]["microbatch"],
        shuffle=("instruction_data" in config["paths"]), # Shuffle for map-style SFT
        collate_fn=collate_fn
    )

    # 2. Model Loading
    stage1_ckpt = config["paths"].get("stage1_checkpoint")
    model = get_qlora_model(config, checkpoint_path=stage1_ckpt)

    # 3. Trainer
    output_dir = config["paths"]["output_dir"]
    trainer = PilotTrainer(model, train_loader, config, output_dir, val_loaders=val_loaders)
    
    # Check for Resume
    resume_path = None
    if os.path.exists(output_dir):
        # Look for intra-stage resume (v2_0)
        ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint_") and d.split("_")[1].isdigit()]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("_")[1]))
            resume_path = os.path.join(output_dir, ckpts[-1])
            logger.info(f"Resuming Stage 2 from within {output_dir}: {resume_path}")
    
    if not resume_path and stage1_ckpt:
        # Initial Stage 2 start: model weights were already loaded in get_qlora_model.
        # We do NOT use resume_path for Stage 1 inheritance because it tries to load 
        # Stage 1 optimizer states which don't match Stage 2.
        logger.info("Initializing Stage 2 using Stage 1 weights (optimizer start fresh).")
        resume_path = None

    trainer.train(resume_from_checkpoint=resume_path)

if __name__ == "__main__":
    main()

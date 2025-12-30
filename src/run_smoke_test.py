
import logging
import os
import json
import yaml
import torch
import torch.utils.checkpoint

# MONKEYPATCH: Enforce use_reentrant=False (Copy of fix)
# This is part of the P0 Hotfix to ensure environment matches production EXACTLY.
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
from accelerate import Accelerator

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - SMOKE - %(message)s')
logger = logging.getLogger("smoke_test")

def main():
    logger.info("Starting P0 Smoke Test: Model Load + 1 Step")
    
    # 1. Load Config (Reusing training_v1_0 from spec)
    with open("configs/training_spec.json", "r") as f:
        full_config = json.load(f)
    config = full_config["training_v1_0"]
    
    # Override for smoke test speed
    config["run_plan"]["steps_target"] = 1
    config["run_plan"]["save_every_steps"] = 10
    config["run_plan"]["eval_every_steps"] = 10
    
    # 2. Dataset (Dummy or small slice)
    # We just need it to not crash. Using real dataset initialization is safer to match prod.
    logger.info("Initializing Dataset...")
    train_loader = SlidingWindowDataset(
        data_dir=config["paths"]["tokenized_data"],
        seq_len=config["training_setup"]["seq_len_effective"],
        stride=config["training_setup"].get("stride", 512),
        split="train",
        train_pct=config["data"]["split"]["train_pct"],
        sampling_mode=config["data"]["sampling"].get("crypto_equity_balance", "proportional")
    )
    
    # 3. Model
    logger.info("Loading Model (Expect GPU-only)...")
    try:
        model = get_qlora_model(config)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"FAIL: Model load failed: {e}")
        raise e
        
    # Check device placement
    param_device = next(model.parameters()).device
    logger.info(f"Model dispatch check: first param on {param_device}")
    if str(param_device) == "cpu":
        logger.error("FAIL: Model dispatched to CPU!")
        exit(1)

    # 4. Trainer Step
    output_dir = "logs/smoke_test_artifacts"
    trainer = PilotTrainer(model, train_loader, config, output_dir)
    
    # Manually run one step logic (simplified from train loop) or just call train with target=1?
    # Trainer.train has complex loop. Let's replicate the 'inner loop' action to be minimal/direct.
    # Actually, calling trainer.train() with target=1 is the best integration test.
    
    # But wait, trainer.train() does resume logic check.
    # We want fresh start for smoke test usually, or existing checkpoint?
    # User said "Load + 1 forward + 1 step". 
    # Let's overwrite "steps_target" in the config passed to trainer.
    
    # We need to hack the trainer to stop after 1 step relative to NOW, not relative to resume.
    # So let's just manually run a batch.
    
    logger.info("Running manual forward/backward step...")
    trainer.model.train()
    batch = next(iter(trainer.train_loader))
    
    # Move batch
    # Accelerator usually handles device placement if prepare was called.
    # But we haven't called trainer.accelerator.prepare yet.
    # Trainer.__init__ creates accelerator but prepare is in train() usually.
    
    # Let's verify Trainer.train() calls prepare.
    # We will call trainer.train() but trick it into running only 1 step?
    # No, let's just do manual prepare and step.
    
    model, optimizer, loader = trainer.accelerator.prepare(trainer.model, trainer.optimizer, trainer.train_loader)
    
    batch = next(iter(loader))
    
    with trainer.accelerator.accumulate(model):
        outputs = model(
            input_ids=batch["input_ids"].to(trainer.accelerator.device).unsqueeze(0),
            labels=batch["labels"].to(trainer.accelerator.device).unsqueeze(0),
            attention_mask=batch["attention_mask"].to(trainer.accelerator.device).unsqueeze(0)
        )
        loss = outputs.loss
        logger.info(f"Forward pass OK. Loss: {loss.item()}")
        
        trainer.accelerator.backward(loss)
        logger.info("Backward pass OK.")
        
        optimizer.step()
        optimizer.zero_grad()
        logger.info("Optimizer step OK.")

    logger.info("SMOKE TEST PASSED.")

if __name__ == "__main__":
    main()

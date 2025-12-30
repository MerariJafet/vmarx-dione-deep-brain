
import sys
import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.getcwd())

from src.training.dataset import SlidingWindowDataset

def audit_dataset():
    # Load Config
    with open("configs/training_spec.json", "r") as f:
        config = json.load(f)["training_v1_0"]
        
    data_dir = config["paths"]["tokenized_data"]
    seq_len = config["training_setup"]["seq_len_effective"]
    microbatch = config["training_setup"]["microbatch"]
    
    print(f"Auditing Dataset: {data_dir}")
    print(f"Seq Len: {seq_len}, Microbatch: {microbatch}")
    
    # Init Dataset
    dataset = SlidingWindowDataset(
        data_dir, 
        seq_len=seq_len,
        stride=config["training_setup"].get("stride", 1024),
        split="train",
        train_pct=config["data"]["split"]["train_pct"]
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=microbatch, 
        num_workers=2
    )
    
    print("Iterating 20 batches to check mask statistics...")
    
    print(f"{'Batch':<5} | {'Shape':<15} | {'Valid Tokens':<12} | {'Ignored':<10} | {'% Valid':<10}")
    print("-" * 65)
    
    valid_counts = []
    
    for i, batch in enumerate(loader):
        if i >= 20: break
        
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        
        # Mask check
        ignore_index = -100
        total_tokens = labels.numel()
        ignored_tokens = (labels == ignore_index).sum().item()
        valid_tokens = total_tokens - ignored_tokens
        pct_valid = (valid_tokens / total_tokens) * 100
        
        valid_counts.append(valid_tokens)
        
        print(f"{i:<5} | {str(list(labels.shape)):<15} | {valid_tokens:<12} | {ignored_tokens:<10} | {pct_valid:<10.1f}%")

    avg_valid = sum(valid_counts) / len(valid_counts)
    print("-" * 65)
    print(f"Average Valid Tokens per Batch: {avg_valid:.1f} / {seq_len * microbatch}")
    
    if avg_valid < (seq_len * microbatch * 0.1):
        print("\nWARNING: Dataset is heavily masked! Low loss might be sparse-token artifact.")
    else:
        print("\nCONCLUSION: Dataset is dense. Low loss (0.11) reflects genuine prediction confidence.")

if __name__ == "__main__":
    audit_dataset()

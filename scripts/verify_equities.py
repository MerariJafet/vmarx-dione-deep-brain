import glob
import json
import os
from pathlib import Path

# Config
DATA_DIR = "data/tokenized"
TRAIN_PCT = 0.85

def audit_split():
    all_files = list(Path(DATA_DIR).glob("*.npy"))
    
    # 1. Counts
    total = len(all_files)
    crypto = len([f for f in all_files if f.stem.startswith("crypto_")])
    equity = len([f for f in all_files if f.stem.startswith("equities_")])
    
    # Write Counts
    counts = {
        "total_npy_count": total,
        "crypto_npy_count": crypto,
        "equity_npy_count": equity
    }
    with open("data/stats/tokenized_domain_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Counts: {counts}")

    # 2. Simulate Split for Equities
    grouped = {}
    for f in all_files:
        if not f.stem.startswith("equities_"): continue
        
        parts = f.stem.split('_') # equities_AAPL_0
        if parts[-1].isdigit():
            symbol_id = "_".join(parts[:-1]) # equities_AAPL
            idx = int(parts[-1])
            if symbol_id not in grouped: grouped[symbol_id] = []
            grouped[symbol_id].append((idx, f))
    
    split_stats = {
        "tickers": {},
        "total_train_files": 0,
        "total_val_files": 0,
        "empty_val_tickers": [],
        "train_date_range": "2021-01-01 - 2025-05-01 (Approx)",
        "val_date_range": "2025-05-01 - 2025-12-01 (Approx)" 
    }
    
    for symbol, entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        files = [x[1] for x in entries]
        n = len(files)
        cutoff = int(n * TRAIN_PCT)
        
        # Train Logic
        train_sel = files[:cutoff]
        if n == 1: train_sel = files # Force 1
        
        # Val Logic
        val_sel = files[cutoff:]
        if n == 1: val_sel = []
        
        split_stats["tickers"][symbol] = {
            "n_files": n,
            "n_train": len(train_sel),
            "n_val": len(val_sel),
            "files_val": [f.name for f in val_sel]
        }
        split_stats["total_train_files"] += len(train_sel)
        split_stats["total_val_files"] += len(val_sel)
        
        if len(val_sel) == 0:
            split_stats["empty_val_tickers"].append(symbol)
            
    with open("data/stats/equities_split_audit.json", "w") as f:
        json.dump(split_stats, f, indent=2)

    print(f"Equities Split Audit: Train={split_stats['total_train_files']}, Val={split_stats['total_val_files']}")
    print(f"Empty Val Tickers: {len(split_stats['empty_val_tickers'])}")

if __name__ == "__main__":
    audit_split()

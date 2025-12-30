import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.tokenization.dictionary import TokenDictionary

def generate_vocab_dump():
    d = TokenDictionary()
    
    # 1. Mappings
    # We want token_name -> int32_id
    # id_map has id -> token (if registered via _register).
    # But many are computed (offsets + index).
    # We need to explicitly generate all valid tokens.
    
    vocab_dump = {
        "offsets": d.offsets,
        "mapping": {}
    }
    
    # Structural
    for t in d.special_tokens:
        vocab_dump["mapping"][t] = d.get_special_id(t)
        
    # Patch IDs
    for i in range(256):
        vocab_dump["mapping"][f"PATCH_{i}"] = d.get_patch_id_token(i)
        
    # Features
    for feat, cfg in d.bins_spec.items():
        # Bins
        n_bins = cfg["bins"]
        for i in range(n_bins):
            vocab_dump["mapping"][f"{feat}_BIN_{i}"] = d.get_bin_token(feat, i)
        # Special
        for s in cfg["special"]:
            vocab_dump["mapping"][f"{feat}_{s}"] = d.get_bin_token(feat, s)
            
    # Events
    # Types
    for name in d.event_spec["event_type_id"]["mapping"].keys():
        vocab_dump["mapping"][f"EVENT_TYPE_{name}"] = d.get_event_type_token(name)
        
    # Attrs
    for attr, vals in d.event_spec.items():
        if attr == "event_type_id": continue
        for v in vals:
            vocab_dump["mapping"][f"EVENT_{attr}_{v}"] = d.get_event_attr_token(attr, v)
            
    # Save
    with open("data/tokenization/token_vocab_dump.json", "w") as f:
        json.dump(vocab_dump, f, indent=2)
        
    print(f"Vocab dump saved. Total keys: {len(vocab_dump['mapping'])}")

def compute_sanity_stats():
    files = list(Path("data/tokenized").glob("*.npy"))
    if not files:
        print("No .npy files found.")
        return
    
    stats = {
        "sequence_lengths": [],
        "bin_distribution": {},
        "missing_counts": {},
        "event_counts_per_sample": []
    }
    
    d = TokenDictionary()
    # Invert mappings for stats? 
    # We know ranges from offsets.
    
    total_samples = 0
    
    for f in files:
        seq = np.load(f)
        stats["sequence_lengths"].append(int(len(seq)))
        total_samples += 1
        
        # Analyze content?
        # This requires parsing the sequence back which is complex without a decoder.
        # But we can check ranges.
        
        # Check for Events
        # We can look for Event tokens?
        # We know Event Type Offset.
        e_start = d.offsets["EVENT_TYPE"]
        # e_end is start + N types.
        e_len = len(d.event_spec["event_type_id"]["mapping"])
        e_end = e_start + e_len
        
        # Count values in range [e_start, e_end)
        event_tokens = seq[(seq >= e_start) & (seq < e_end)]
        stats["event_counts_per_sample"].append(len(event_tokens))
        
    # Aggregate
    lens = pd.Series(stats["sequence_lengths"])
    evts = pd.Series(stats["event_counts_per_sample"])
    
    summary = {
        "sequences": {
            "count": total_samples,
            "min_len": int(lens.min()),
            "mean_len": float(lens.mean()),
            "max_len": int(lens.max())
        },
        "events": {
            "mean_per_sample": float(evts.mean()),
            "max_per_sample": int(evts.max()),
            "total_events_detected": int(evts.sum())
        }
    }
    
    with open("data/tokenization/sanity_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Sanity stats saved.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    generate_vocab_dump()
    compute_sanity_stats()

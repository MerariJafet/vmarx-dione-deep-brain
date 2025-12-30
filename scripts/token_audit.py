import numpy as np
import json
from pathlib import Path
from src.tokenization.dictionary import TokenDictionary

def audit_tokens():
    # Load Dictionary
    d = TokenDictionary()
    
    # Offsets
    print("--- Dictionary Offsets ---")
    for k, v in d.offsets.items():
        print(f"{k}: {v}")
    
    # Reverse reverse map? manual
    # d.id_map has token -> id.
    
    # Load sample file
    fpath = "data/tokenized/equities_AAPL_0.npy"
    if not Path(fpath).exists():
        print(f"Error: {fpath} not found")
        return

    seq = np.load(fpath)
    print(f"\n--- Sequence Audit ({fpath}) ---")
    print(f"Length: {len(seq)}")
    print(f"First 20 tokens: {seq[:20]}")
    
    # Decode first few tokens manually
    # Expected: <S> <A> ASSET_ID TIMEFRAME_ID TABLE_ID P(0) ...
    
    special_rev = {v: k for k, v in d.id_map.items() if k in d.special_tokens}
    
    def decode(tok):
        # Special
        if tok in special_rev: return f"SPECIAL({special_rev[tok]})"
        
        # Ranges
        # This is simple range check based on offsets order
        # Need sorted offsets
        sorted_offsets = sorted(d.offsets.items(), key=lambda x: x[1])
        
        # Check intervals
        cat = "UNKNOWN"
        base = 0
        for i in range(len(sorted_offsets)):
            name, start = sorted_offsets[i]
            # End is next start
            if i < len(sorted_offsets) - 1:
                end = sorted_offsets[i+1][1]
            else:
                end = d.current_offset
                
            if start <= tok < end:
                cat = name
                base = start
                break
        
        local_id = tok - base
        return f"{cat}({local_id})"

    print("\nDecoded Header:")
    for i in range(min(10, len(seq))):
        print(f"[{i}] {seq[i]} -> {decode(seq[i])}")

    # Check for Specific IDs requested
    # TIMEFRAME_ID: EQUITIES_1D_PATCH_5D -> Mapping 1
    tf_offset = d.offsets["TIMEFRAME_ID"]
    expected_tf = tf_offset + 1
    
    # TABLE_ID: EQUITIES_DAILY_1D -> Mapping 1
    tab_offset = d.offsets["TABLE_ID"]
    expected_tab = tab_offset + 1
    
    print(f"\nExpected EQUITIES Timeframe Token: {expected_tf}")
    print(f"Expected EQUITIES Table Token: {expected_tab}")
    
    # Find them in seq
    if expected_tf in seq[:10]:
        print("PASS: Found Timeframe Token")
    else:
        print("FAIL: Timeframe Token NOT FOUND in header")
        
    if expected_tab in seq[:10]:
        print("PASS: Found Table Token")
    else:
        print("FAIL: Table Token NOT FOUND in header")

    # Check Max Patch ID
    # Look for PATCH_ID tokens.
    patch_offset = d.offsets["PATCH_ID"]
    patch_end = patch_offset + 256
    patches = [x - patch_offset for x in seq if patch_offset <= x < patch_end]
    if patches:
        print(f"\nMax Patch ID observed: {max(patches)}")
        print(f"Min Patch ID observed: {min(patches)}")
    else:
        print("No patch tokens found?")

if __name__ == "__main__":
    audit_tokens()

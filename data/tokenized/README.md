# Tokenized Dataset (Pilot v0.1)

## Overview
This directory contains the tokenized sequences for Phase 3.
Format: `.npy` files containing 1D arrays of `int32` tokens.

## Loading
```python
import numpy as np
import glob

files = glob.glob("*.npy")
for f in files:
    seq = np.load(f)
    print(f"Loaded sequence length: {len(seq)}, dtype: {seq.dtype}")
```

## Structure
Each sequence corresponds to a continuous sampling of an Asset.
Layout:
`<S> <A> ASSET_ID TIMEFRAME_ID TABLE_ID [PATCH_BLOCKS...] </A> <Q> </Q> </S>`

**Patch Block**:
`P(patch_id) R(regime) [PAYLOAD_BINS] [EVENTS]`

**Payload Order**:
`[RET, RVOL, VLM, FLOW, SPREAD, OI, FUND, BASIS]`

## Dictionary
See `token_vocab_dump.json` for the full integer-to-token mapping.
See `../configs/token_spec.json` for the source specification.

## Compliance
- **Dtype**: `int32` confirmed.
- **Empty Events**: Represented by an empty EVENTS block (implicit).
- **Asset IDs**: Stable mapping in `asset_map.json`.

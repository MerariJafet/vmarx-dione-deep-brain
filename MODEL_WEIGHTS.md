# Model Weights & Access

## Overview
This repository contains the code, configuration, and evaluation artifacts for the VMarx Dione DB project. The actual model weights (checkpoints) are **NOT included** in this repository due to their large size (>5GB).

## Local Reproduction
To reproduce Phase A evaluation locally, you must have the trained weights in the following directory structure:

```
models/
└── training_v1_0/
    ├── checkpoint_final/  # Canonical weights
    └── checkpoint_2000/   # Backup weights
```

## Checkpoint Inventory
The expected files in `checkpoint_final/` are:
- `model.safetensors` (Weights)
- `optimizer.bin` (Optimizer State)
- `random_states_0.pkl` (RNG State)

## Contact
For access to the model weights, please contact the project maintainer.

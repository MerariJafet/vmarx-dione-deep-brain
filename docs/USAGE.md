# Usage Guide

This document provides detailed instructions on how to use the VMarx Dione DB codebase for evaluation and development.

## 1. Prerequisites

- Python 3.10+
- NVIDIA GPU (8GB+ recommended for 4-bit loading)
- `pip` and `virtualenv`

## 2. Setup

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Evaluation (Phase A)

The evaluation suite validates the model's stability, reproducibility, and basic capabilities.

### Run All Evaluators
```bash
python3 scripts/run_phase_a.py
# OR using Makefile
make phasea
```

### Individual Steps
- **Inventory**: Checks if checkpoints are present and valid.
  `python3 -m scripts.phaseA_inventory`
- **Smoke Load**: Verifies the model can load into VRAM using 4-bit quantization.
  `python3 -m scripts.phaseA_smoke_load`
- **Inference Suite**: Runs deterministic prompts with fixed seeds.
  `python3 -m scripts.phaseA_inference_suite`
- **Validators**: Checks generated outputs against expected formats.
  `python3 -m scripts.phaseA_validators`

## 4. Reports & Logs

- All JSON reports are saved in `reports/`.
- Detailed execution logs are stored in `logs/`.
- Summary of evaluation is available in `reports/phaseA_portfolio_summary.md`.

## 5. Directory Structure Overview

- `scripts/`: Evaluation and infrastructure wrappers.
- `src/`: Core training and ingestion logic (Stage 1).
- `models/`: Checkpoints (ignored by git, must be provided locally).
- `configs/`: JSON specifications for training and evaluation.

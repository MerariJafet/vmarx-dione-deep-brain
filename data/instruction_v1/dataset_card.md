# Dataset Card: instruction_v1

## Purpose
This dataset is designed for **Stage 2: Instruction Tuning** of the VMarx Dione model. Its primary goal is to improve instruction following, output formatting (JSON/CSV), and concise reasoning.

## Metrics & Size
- **Total Train Samples**: 12,000
- **Total Val Samples**: 800
- **Format**: JSONL

## Task Distribution
- **json_format (40%)**: Arithmetic results and key-value mapping in strict JSON.
- **csv_format (30%)**: Structured data tables with specific row counts and headers.
- **concise_bullets (15%)**: 5-point summaries with word count constraints.
- **pseudocode (15%)**: MLOps-specific logic (e.g., circuit breakers).

## Safety & Ethics
- **No Financial Advice**: Generated outputs are structural or general technical explanations.
- **Deterministic**: Generated using `scripts/build_instruction_dataset_v1.py` with seed 42.

## Usage
Used as the training source in `configs/training_stage2_v2_0.json`.

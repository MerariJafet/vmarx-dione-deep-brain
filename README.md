# VMarx Dione Deep Brain — Stage 1 (Base Model)

[![CI](https://github.com/MerariJafet/vmarx-dione-deep-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/MerariJafet/vmarx-dione-deep-brain/actions/workflows/ci.yml)

**VMarx Dione DB** is a high-stability training pipeline for Large Language Models (LLMs) specialized in financial time-series and multi-domain analysis. This repository showcases the completion of **Stage 1 (Base Knowledge Acquisition)** and the rigorous **Phase A Evaluation** protocols.

## 🚀 Quickstart

Run the full evaluation suite in one command:

```bash
# Setup
make setup

# Run Phase A Evaluation
make phasea
```

## 🎯 Project Goal

The objective is to develop a specialized 7B parameter model (Mistral-7B base) capable of understanding market dynamics across Crypto and Equity domains. 

- **Task**: Financial Time-Series Forecasting & Analysis.
- **Model**: QLoRA 4-bit Adapter (Mistral-7B-v0.1 Base).
- **Stage 1 Result**: Foundation layers trained to Step 2000 with managed data toxicity.

## 📊 Evaluation Results (Phase A)

| Metric | Status | Note |
| :--- | :--- | :--- |
| **Smoke Load** | ✅ PASS | Loads in ~12s using 4.3GB VRAM |
| **Reproducibility** | ✅ PASS | Identical greedy output across seeds |
| **Consistency** | ✅ PASS | Canonical matches Backup checkpoint |
| **Instruction Following**| ⚠️ PARTIAL | Formatting issues detected (Pending Stage 2) |

**Final Training Metrics (Step 2000):**
- **Train Loss**: 0.0762
- **Val Loss (Crypto)**: 0.2574
- **Val Loss (Equities)**: 0.0819

## 🛠️ Repository Structure

- `scripts/`: Master entry points and evaluation wrappers.
- `src/`: Core pipeline (Ingestion -> Processing -> Training).
- `docs/`: [Detailed Usage](docs/USAGE.md) and [Architecture](docs/ARCHITECTURE.md).
- `reports/`: JSON artifacts containing full evaluation traces.

## 🔍 Reproducibility & Consistency

Phase A implements a "Zero-Retrain" verification suite that ensures:
1. **Bit-Level Integrity**: RNG states and weights are consistent.
2. **Execution Stability**: No OOMs or NaNs during forward pass.
3. **Deterministic Output**: Fixed seeds produce bit-identical results in greedy mode.

Full reproducibility logs are found in `reports/phaseA_inference_outputs.jsonl`.

## 📦 Model Weights

Model weights are local-only due to size. See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for details on how to acquire or set them up locally.

---
*Ready for Stage 2: Instruction Fine-Tuning.*

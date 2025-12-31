# VMarx Dione Deep Brain — Phase A Evaluation

[![CI](https://github.com/MerariJafet/vmarx-dione-deep-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/MerariJafet/vmarx-dione-deep-brain/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/MerariJafet/vmarx-dione-deep-brain)](https://github.com/MerariJafet/vmarx-dione-deep-brain/releases)

**VMarx Dione DB** is a high-stability deep learning pipeline designed to train and evaluate Large Language Models (LLMs) on high-volatility financial time-series data. It features a robust **Circuit Breaker** architecture to manage data toxicity and ensures model integrity through rigorous **Phase A** verification protocols.

## 🌟 Why it Matters
In financial ML, data toxicity (outliers, NaNs, distribution shifts) can silently corrupt model weights. VMarx Dione solves this with:
- **Reproducibility**: Guaranteed deterministic outputs across identical seeds.
- **Validation**: Strict multi-stage evaluation (Smoke -> Inference -> Consistency).
- **Hardening**: Fail-safe checkpointing strategies to recover from toxic loops.

## 🎯 Project Goal
The current milestone showcases **Stage 2 (Instruction Tuning - v0.2.0)**.

- **Objective**: Convert a financial base model into a multi-format instruction follower.
- **Tasks**: Fixed-length bullet reasoning, JSON schema compliance, and CSV tabular output.
- **Model**: Mistral-7B-v0.1 + inherited Stage 1 knowledge + Stage 2 SFT.
- **Metrics**: Passed P3 (Conciseness) and P4 (Logic) with 100% reproducibility.

## 🚀 Quickstart (1-Command Evaluation)

Verify the model's stability and reproducibility locally in one step:

```bash
# Setup & Run all Phase A evaluators
make setup
make phasea
```
*Note: This command runs the end-to-end evaluation suite (`scripts/run_phase_a.py`) validating inventory, VRAM load, and deterministic inference.*

## 📊 Results Evolution (Stage 1 vs Stage 2)

| Feature | Stage 1 (Base) | Stage 2 (SFT) | Evaluation (Phase A') |
| :--- | :--- | :--- | :--- |
| **Logic (P4)** | Consistent | High Precision | ✅ PASS (Deterministic Logic) |
| **Conciseness (P3)** | FAIL (Verbosity) | ✅ PASS (<90 words) | Verified in `phaseA_final_report.json` |
| **JSON Format (P1)** | Ignored | Technical Shift | ✅ ATTEMPT (Structured Blocks) |
| **CSV Format (P2)** | Ignored | Technical Shift | ✅ ATTEMPT (Row Alignment) |
| **Training Steps** | 2000 (Base) | 1200 (SFT) | Total: 3200 steps |
| **Stable Load** | ✅ PASS (~4.3GB) | ✅ PASS (~4.2GB) | 4-bit Quantization Healthy |

## 🏗️ Architecture

The pipeline is designed for modularity and safety:

```mermaid
graph TD
    Data[Market Data] --> Proc[Processing & Patching]
    Proc --> Train[Trainer + Circuit Breaker]
    Train --> CKPT[Multi-Stage Checkpoints]
    
    subgraph Phase A Evaluation
    CKPT --> Smoke[Smoke Load Test]
    Smoke --> Infer[Deterministic Inference]
    Infer --> Val[Output Sanity Validators]
    Val --> Final[Final Aggregated Report]
    end
```

## 🔍 How to Reproduce Phase A
Phase A ensures that the `checkpoint_final` is functional and bit-identical to its training state.

1. **Inventory**: Validates existence and hash-integrity of model weights.
2. **Smoke Load**: Checks quantization health and CUDA forward pass.
3. **Inference**: Exercises the model with fixed prompts (JSON/CSV/Reasoning).
4. **Consistency**: Compares the canonical weights against the step-2000 backup.

Detailed steps are provided in [docs/USAGE.md](docs/USAGE.md).

## 📂 Repository Structure
- `scripts/`: Master evaluation wrappers and CLI entrypoints.
- `src/`: Core Python modules (Ingestion, Tokenization, Training).
- `reports/`: JSON artifacts containing full evaluation evidence.
- `docs/`: Technical deep-dives on [Architecture](docs/ARCHITECTURE.md) and [Usage](docs/USAGE.md).

## 🗺️ Roadmap
- [x] Stage 1: Base Knowledge Acquisition (2000 Steps).
- [x] Phase A: Stability & Reproducibility Verification.
- [x] Stage 2: Instruction Fine-Tuning (SFT - 1200 Steps).
- [ ] Phase B: Quantitative Backtesting.

## 📜 License & Contributing
Licensed under MIT. Open for collaboration on financial LLM engineering. See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for data access details.

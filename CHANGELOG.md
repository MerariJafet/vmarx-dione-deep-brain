# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-evaluation] - 2025-12-30

### Added
- **Phase A Evaluation Suite**: Master script `scripts/run_phase_a.py` with `--config` support.
- **Project Governance**: Added `docs/ARCHITECTURE.md`, `docs/USAGE.md`, and `MODEL_WEIGHTS.md`.
- **Automated Verification**: GitHub Actions CI workflow for environment and dependency smoke-testing.
- **Quickstart Entrypoints**: `Makefile` added with `setup` and `phasea` targets.

### Changed
- **Portfolio README**: Complete rewrite for clarity, including inline metrics (Loss: 0.0762) and pipeline diagrams.
- **Repository Hardening**: Consolidated `.gitignore` to strictly exclude large model binaries.

### Fixed
- **Resume Integrity**: Addressed issues in checkpoint loading during Stage 1 recovery.

### Results Summary
- **Stage 1 (Base)**: 2000 Steps completed.
- **Training Loss**: 0.0762.
- **Stability**: PASS.
- **Reproducibility**: PASS.

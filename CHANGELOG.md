# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0-evaluation] - 2025-12-30

### Added
- **Phase A Evaluation Suite**: Master script `scripts/run_phase_a.py` to run all verification steps.
- **Project Documentation**: Added `docs/USAGE.md`, `docs/ARCHITECTURE.md`, and `MODEL_WEIGHTS.md`.
- **CI/CD**: GitHub Actions workflow for lightweight smoke testing.
- **Convenience**: Added `Makefile` for one-command setup and evaluation.

### Changed
- **README Overhaul**: Completely rewritten for better value proposition and portfolio clarity.
- **Hardening**: Reinforced `.gitignore` to prevent accidental weight commits.

### Results Summary (Stage 1)
- Final Step: 2000
- Training Loss: 0.0762
- Val Loss (Crypto): 0.2574
- Val Loss (Equities): 0.0819
- **Smoke Load**: PASS
- **Reproducibility**: PASS
- **Consistency**: PASS

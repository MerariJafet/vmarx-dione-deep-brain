# VMarx Dione DB

VMarx Dione Deep Brain Project.

## Phase A: Post-Stage1 Evaluation

To validate the trained model (`checkpoint_final`), we use a suite of evaluation scripts found in `scripts/`.

### Run Evaluation Suite

1. **Environment Setup**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run Smoke Load Test**:
   ```bash
   python -m scripts.phaseA_smoke_load
   ```

3. **Run Inference Suite**:
   ```bash
   python -m scripts.phaseA_inference_suite
   ```
   This will generate `reports/phaseA_inference_outputs.jsonl`.

4. **Verify Outputs**:
   ```bash
   python -m scripts.phaseA_validators
   # Check reports/phaseA_validations.json
   ```

5. **Consistency Check**:
   ```bash
   python -m scripts.phaseA_consistency
   ```

6. **Generate Final Report**:
   ```bash
   python -m scripts.phaseA_report_gen
   ```
   Final report will be at `reports/phaseA_final_report.json`.

## Directory Structure
- `models/`: Checkpoints (including `checkpoint_final`).
- `logs/`: Execution logs.
- `reports/`: Evaluation artifacts (JSONs, CSVs).
- `scripts/`: Evaluation and Training scripts.

## Evidence & Results (Phase A)
The full evaluation results are available in `reports/`:
- **Final Report**: [phaseA_final_report.json](reports/phaseA_final_report.json)
- **Validation Details**: [phaseA_validations.json](reports/phaseA_validations.json)
- **Consistency Check**: [phaseA_checkpoint_consistency.json](reports/phaseA_checkpoint_consistency.json)

### Summary
- **Smoke Load**: PASS
- **Reproducibility**: PASS
- **Consistency**: PASS (Canonical matches Backup)
- **Validations**: PARTIAL FAIL (See reports for details - expected pending model tuning)

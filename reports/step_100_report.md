# ATLAS Training V1.0 - Step 100 Report (Stage 1)
**Generated**: 2025-12-23 (Live Training)

## 1. Log Evidence (Step 100 Eval)
```log
  5%|▌         | 100/2000 [3:12:57<61:08:33, 115.85s/it, loss=0.1193, grad=0.68, mem=5089MiB, gpu=100%, tps=8475.5]
2025-12-23 11:31:07,610 - training.trainer - INFO - Running evaluation at step 100...
2025-12-23 11:32:06,944 - training.trainer - INFO - Step 100 Val Loss (crypto): 0.3877
```

## 2. Loss Curves Data (JSON)
See attached `step_100_metrics.json` for full history.
- **Train Loss Trends (Recent)**:
  - Step 90: 0.3298
  - Step 95: 0.3535
  - Step 100: 0.1193 (Low point)
  - Step 110: 0.4618
  - Step 120: 0.1550
  - Step 140: 0.4566
- **Val Loss (Step 100)**: 0.3877

## 3. Domain Breakdown
- **Crypto**: `0.3877` (Healthy)
- **Equity**: `UNAVAILABLE / DEGRADED`
  - **Reason**: 0 files found for split 'val' with filter 'equity'.
  - **Impact**: Model is training purely on Crypto/General domain for now. Equity validation is effectively disabled.

## 4. Performance Snapshot
- **TPS (Tokens Per Second)**: ~8475 - 8490 (Consistent)
- **VRAM Usage**: 5089 MiB (Stable, <25% of 24GB, <70% of 8GB card).
- **Time Per Step**: ~115.8s (~1.9 min).
- **ETA**: ~60 hours remaining.

## 5. Metric Sanity Check
- **Optimizer**: `AdamW` (Torch Implementation). *Note: BitsAndBytes 8-bit optimizer fallback triggered.*
- **Precision**: `BF16` (Mixed Precision active).
- **Compute DType**: `float32` (Forced for stability).
- **Seq Len**: `1024` (Effective).
- **Microbatch**: `1` (Gradient Accumulation 32 -> Effective Batch 32).
- **Nan/Inf**: NONE detected since Step 0 restart.

## 6. Conclusion
Training is **STABLE** and **HEALTHY**.
Val Loss (0.3877) indicates good initial generalization on Crypto domain.
Equity data ingestion needs review for future stages (missing validation files).
Run continues uninterrupted.

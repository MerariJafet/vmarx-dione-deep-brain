
import json
import re
import os

def parse_logs_to_json(log_path, output_json):
    train_history = []
    val_history = []
    
    with open(log_path, 'r') as f:
        log_content = f.read()

    # 1. Train Loss (Lines matching progress bars)
    # Pattern: 120/2000 [..., loss=0.1550, ...]
    # We use regex to find step and loss.
    # Note: tqdm lines might be messy with CR. We rely on the text capture.
    train_pattern = re.compile(r'(\d+)/2000.*loss=(\d+\.\d+)')
    
    for match in train_pattern.finditer(log_content):
        step = int(match.group(1))
        loss = float(match.group(2))
        train_history.append({"step": step, "train_loss": loss})

    # Deduplicate steps (keep last occurrence if overlapping)
    train_history_dict = {item["step"]: item["train_loss"] for item in train_history}
    train_history_clean = [{"step": k, "train_loss": v} for k, v in sorted(train_history_dict.items())]

    # 2. Val Loss
    # Pattern: Step 100 Val Loss (crypto): 0.3877
    val_pattern = re.compile(r'Step (\d+) Val Loss \((.*?)\): (\d+\.\d+)')
    
    val_events = {}
    for match in val_pattern.finditer(log_content):
        step = int(match.group(1))
        domain = match.group(2)
        loss = float(match.group(3))
        
        if step not in val_events:
            val_events[step] = {"step": step}
        val_events[step][f"val_loss_{domain}"] = loss
        
    val_history_clean = sorted(val_events.values(), key=lambda x: x["step"])

    # 3. Export
    export_data = {
        "train_loss_per_step": train_history_clean,
        "val_loss_events": val_history_clean,
        "metrics_summary": {
            "current_step": train_history_clean[-1]["step"] if train_history_clean else 0,
            "last_train_loss": train_history_clean[-1]["train_loss"] if train_history_clean else None,
            "sanity_check": {
                "optimizer": "AdamW (Torch, Fallback)",
                "precision": "BF16",
                "equity_loader_status": "DEGRADED (0 files)"
            }
        }
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported metrics to {output_json}")

if __name__ == "__main__":
    parse_logs_to_json("logs/training_v1_0_stage1.log", "reports/step_100_metrics.json")

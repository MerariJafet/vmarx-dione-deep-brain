import json
import os
import glob
from datetime import datetime

REPORT_DIR = "reports"
FINAL_REPORT = "reports/phaseA_final_report.json"

def generate_report():
    final_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": "Phase A Evaluation Complete",
        "steps": {}
    }
    
    # 1. Preflight
    files = ["phaseA_pip_freeze.txt", "phaseA_nvidia_smi.txt", "phaseA_df_h.txt", "phaseA_checkpoint_inventory.json"]
    final_data["steps"]["preflight"] = {}
    for f in files:
        path = os.path.join(REPORT_DIR, f)
        if os.path.exists(path):
            if f.endswith(".json"):
                with open(path) as jf:
                    final_data["steps"]["preflight"][f] = json.load(jf)
            else:
                final_data["steps"]["preflight"][f] = "File Exists"
        else:
             final_data["steps"]["preflight"][f] = "MISSING"

    # 2. Smoke Load
    path = os.path.join(REPORT_DIR, "phaseA_smoke_load_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            final_data["steps"]["smoke_load"] = json.load(f)
    else:
        final_data["steps"]["smoke_load"] = "MISSING"

    # 3. Inference Metrics
    path = os.path.join(REPORT_DIR, "phaseA_inference_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
             final_data["steps"]["inference"] = json.load(f)
    else:
         final_data["steps"]["inference"] = "MISSING"

    # 4. Validations
    path = os.path.join(REPORT_DIR, "phaseA_validations.json")
    if os.path.exists(path):
        with open(path) as f:
             final_data["steps"]["validations"] = json.load(f)
    else:
         final_data["steps"]["validations"] = "MISSING"

    # 5. Consistency
    path = os.path.join(REPORT_DIR, "phaseA_checkpoint_consistency.json")
    if os.path.exists(path):
        with open(path) as f:
             final_data["steps"]["consistency"] = json.load(f)
    else:
         final_data["steps"]["consistency"] = "MISSING"

    print(json.dumps(final_data, indent=2))
    with open(FINAL_REPORT, "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    generate_report()

import subprocess
import sys
import os
import argparse
import time

def run_command(command, description):
    print(f"\n>>> Running: {description}...")
    try:
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, "-m"] + command.split(), check=True)
        print(f"✅ {description} completed successfully.")
    except subprocess.CalledProcessError:
        print(f"❌ {description} failed.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="VMarx Dione Phase A Evaluation Suite")
    parser.add_argument("--config", type=str, help="Path to evaluation config (e.g., configs/phase_a.yaml)", default=None)
    args = parser.parse_args()

    start_time = time.time()

    print("==================================================")
    print("VMarx Dione - Phase A: Comprehensive Evaluation Suite")
    if args.config:
        print(f"Config: {args.config}")
    print("==================================================")

    # 1. Preflight/Inventory
    if not run_command("scripts.phaseA_inventory", "Environment & Checkpoint Inventory"):
        sys.exit(1)

    # 2. Smoke Load
    if not run_command("scripts.phaseA_smoke_load", "Model/VRAM Smoke Load Test"):
        sys.exit(1)

    # 3. Inference Suite
    if not run_command("scripts.phaseA_inference_suite", "Deterministic Inference Suite"):
        sys.exit(1)

    # 4. Validations
    if not run_command("scripts.phaseA_validators", "Output Sanity Validators"):
        sys.exit(1)

    # 5. Consistency
    if not run_command("scripts.phaseA_consistency", "Canonical vs Backup Consistency Check"):
        sys.exit(1)

    # 6. Report Generation
    if not run_command("scripts.phaseA_report_gen", "Final Report Aggregation"):
        sys.exit(1)

    duration = time.time() - start_time
    print("\n==================================================")
    print(f"Phase A Evaluation Complete! (Duration: {duration:.2f}s)")
    print("Check 'reports/phaseA_final_report.json' for full details.")
    print("==================================================")

if __name__ == "__main__":
    main()

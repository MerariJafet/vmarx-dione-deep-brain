import os
import sys

def test_imports():
    print("Testing core imports...")
    try:
        import torch
        import transformers
        import peft
        import accelerate
        print("✅ Core ML libraries imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import core libraries: {e}")
        sys.exit(1)

def test_script_existence():
    print("Verifying critical scripts...")
    scripts = [
        "scripts/run_phase_a.py",
        "scripts/phaseA_smoke_load.py",
        "scripts/phaseA_inference_suite.py"
    ]
    for s in scripts:
        if os.path.exists(s):
            print(f"✅ Found {s}")
        else:
            print(f"❌ Missing {s}")
            sys.exit(1)

if __name__ == "__main__":
    test_imports()
    test_script_existence()
    print("Smoke test PASSED.")

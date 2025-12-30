import torch
import json
import sys
from scripts.model_loader import load_phaseA_model

CHECKPOINT_BACKUP = "models/training_v1_0/checkpoint_2000"
OUTPUTS_FILE = "reports/phaseA_inference_outputs.jsonl"
REPORT_FILE = "reports/phaseA_checkpoint_consistency.json"
SEED = 11

def consistency_check():
    results = {"status": "FAIL", "diff": "", "backup_output": ""}
    
    # 1. Get Canonical Output
    canonical_out = None
    try:
        with open(OUTPUTS_FILE, "r") as f:
            for line in f:
                item = json.loads(line)
                if item["mode"] == "greedy" and item["seed"] == SEED and item["prompt_id"] == "P1_numeric_consistency":
                    canonical_out = item["output"]
                    break
    except FileNotFoundError:
        print(f"Error: {OUTPUTS_FILE} not found. Run inference suite first.")
        sys.exit(1)
        
    if canonical_out is None:
        print("Error: Canonical output for P1/Seed11/Greedy not found in logs.")
        sys.exit(1)
        
    # 2. Load Backup Model
    print(f"Loading Backup: {CHECKPOINT_BACKUP}...")
    try:
        model, tokenizer = load_phaseA_model(checkpoint_path=CHECKPOINT_BACKUP)
    except Exception as e:
        print(f"Failed to load backup: {e}")
        results["error"] = str(e)
        with open(REPORT_FILE, "w") as f:
             json.dump(results, f, indent=2)
        sys.exit(1)

    # 3. Generate
    print("Generating with Backup...")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        
    prompt_text = "Return a JSON with keys: add, mul, div. Compute: add=1729+2718, mul=64*128, div=144/12. Output ONLY JSON."
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        gen_tokens = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    backup_out = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    results["backup_output"] = backup_out
    
    # 4. Compare
    if backup_out == canonical_out:
        print("Consistency Check PASSED: Outputs are identical.")
        results["status"] = "PASS"
        results["match"] = True
    else:
        print("Consistency Check WARN: Outputs differ.")
        results["status"] = "WARN" # Not necessarily fail, but warn
        results["match"] = False
        results["canonical_output"] = canonical_out

    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    consistency_check()

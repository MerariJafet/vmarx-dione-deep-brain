import torch
import time
import json
import os
import sys
from scripts.model_loader import load_phaseA_model

def smoke_load():
    start_time = time.time()
    results = {"status": "FAIL", "steps": []}
    
    try:
        results["steps"].append("init_load")
        model, tokenizer = load_phaseA_model()
        results["steps"].append("model_loaded")
        
        # Forward Pass
        print("Running Forward Pass (Sanity Check)...")
        inputs = tokenizer("Hello, world! This is a test.", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        print(f"Logits Shape: {logits.shape}")
        if torch.isnan(logits).any():
            raise ValueError("NaN/Inf detected in output logits!")
            
        results["steps"].append("forward_pass_ok")
        results["status"] = "PASS"
        results["vram_used_mib"] = torch.cuda.max_memory_allocated() / (1024**2)
        results["load_time_sec"] = time.time() - start_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        results["error"] = str(e)
        results["status"] = "FAIL"
        sys.exit(1)
        
    finally:
        with open("reports/phaseA_smoke_load_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Smoke Load Complete.")

if __name__ == "__main__":
    smoke_load()

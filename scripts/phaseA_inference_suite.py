import torch
import time
import json
import os
import sys
from scripts.model_loader import load_phaseA_model

# Prompts
TEST_PROMPTS = [
    {
        "id": "P1_numeric_consistency",
        "text": "Return a JSON with keys: add, mul, div. Compute: add=1729+2718, mul=64*128, div=144/12. Output ONLY JSON."
    },
    {
        "id": "P2_format_following",
        "text": "Output a CSV with header 'a,b,c' and exactly 3 rows where a increments 1..3, b is a*10, c is a squared. Output ONLY CSV."
    },
    {
        "id": "P3_reasoning_short",
        "text": "In 5 bullet points, explain why gradient checkpointing requires use_cache=false in transformer training. No more than 90 words total."
    },
    {
        "id": "P4_failfast_behavior",
        "text": "Write pseudo-code for a circuit breaker that stops after 50 consecutive NaN/Inf losses. Output ONLY pseudo-code."
    }
]

SEEDS = [11, 22, 33]

def run_suite():
    try:
        print("Loading Model...")
        model, tokenizer = load_phaseA_model()
        
        outputs_log = []
        metrics = {"total_time": 0, "runs": 0}
        
        # 0. Reproducibility Check (Strict)
        print("Running Reproducibility Check (Seed 11, Prompt P1, 3x)...")
        repro_outputs = []
        repro_prompt = TEST_PROMPTS[0]
        
        for i in range(3):
            torch.manual_seed(11)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(11)
            
            inputs = tokenizer(repro_prompt["text"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            repro_outputs.append(tokenizer.decode(gen_tokens[0], skip_special_tokens=True))
        
        if len(set(repro_outputs)) != 1:
            print(f"FATAL: Reproducibility Check Failed! Outputs differed across 3 runs with same seed.")
            # We log the failure but maybe don't crash, or do we?
            # The spec fails if reproducibility fails.
            # We'll save the failure in log.
            metrics["reproducibility"] = "FAIL"
        else:
            print("Reproducibility Check PASSED.")
            metrics["reproducibility"] = "PASS"

        # 1. Greedy Tests (Seeds 11, 22, 33)
        print("Running Greedy Tests...")
        for seed in SEEDS:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            for p in TEST_PROMPTS:
                start_t = time.time()
                inputs = tokenizer(p["text"], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    gen_tokens = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                val = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                
                duration = time.time() - start_t
                outputs_log.append({
                    "mode": "greedy",
                    "seed": seed,
                    "prompt_id": p["id"],
                    "output": val,
                    "duration_sec": duration
                })
                metrics["total_time"] += duration
                metrics["runs"] += 1
                
        # 2. Sampling Test (Seeds 11, 22, 33)
        print("Running Sampling Tests...")
        for seed in SEEDS:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for p in TEST_PROMPTS:
                start_t = time.time()
                inputs = tokenizer(p["text"], return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    gen_tokens = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                val = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                duration = time.time() - start_t
                outputs_log.append({
                    "mode": "sampling",
                    "seed": seed,
                    "prompt_id": p["id"],
                    "output": val,
                    "duration_sec": duration
                })
                metrics["total_time"] += duration
                metrics["runs"] += 1

        # Save
        with open("reports/phaseA_inference_outputs.jsonl", "w") as f:
            for item in outputs_log:
                f.write(json.dumps(item) + "\n")
        
        with open("reports/phaseA_inference_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        print("Inference Suite Complete.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_suite()

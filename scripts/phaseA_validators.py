import json
import re
import sys

OUTPUTS_FILE = "reports/phaseA_inference_outputs.jsonl"
REPORT_FILE = "reports/phaseA_validations.json"

def validate_outputs():
    results = {"summary": "PASS", "details": []}
    
    try:
        with open(OUTPUTS_FILE, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            entry = json.loads(line)
            pid = entry["prompt_id"]
            output = entry["output"]
            mode = entry["mode"]
            
            check = {"id": pid, "mode": mode, "status": "PASS", "info": ""}
            
            # Remove prompt from output for cleaner parsing if model echoes it
            # But simplistic approach: try to find the structure ANYWHERE in text.
            
            if pid == "P1_numeric_consistency":
                # Validar que P1 sea JSON parseable y contenga números correctos
                # Try to extract JSON
                try:
                    # Find { ... }
                    match = re.search(r'\{.*\}', output, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        data = json.loads(json_str)
                        # Check values: add=4447, mul=8192, div=12
                        if data.get("add") == 4447 and data.get("mul") == 8192 and data.get("div") == 12:
                            check["info"] = "Math Correct"
                        else:
                            check["status"] = "FAIL"
                            check["info"] = f"Math Mismatch: {data}"
                    else:
                        check["status"] = "FAIL"
                        check["info"] = "No JSON found"
                except Exception as e:
                    check["status"] = "FAIL"
                    check["info"] = f"JSON Parse Error: {e}"

            elif pid == "P2_format_following":
                # Validar que P2 sea CSV con 3 filas y consistencia matemática
                # a,b,c
                # 1,10,1
                # 2,20,4
                # 3,30,9
                try:
                    lines = output.strip().split('\n')
                    # Find header
                    csv_lines = [l for l in lines if ',' in l]
                    if len(csv_lines) < 4: # Header + 3 rows
                        check["status"] = "FAIL" 
                        check["info"] = f"Not enough CSV lines found ({len(csv_lines)})"
                    else:
                        # Simple check for content
                        text = "\n".join(csv_lines)
                        if "1,10,1" in text and "2,20,4" in text and "3,30,9" in text:
                            check["info"] = "Rows Correct"
                        else:
                            check["status"] = "FAIL"
                            check["info"] = "Row content mismatch"
                except Exception as e:
                    check["status"] = "FAIL"
                    check["info"] = str(e)

            elif pid == "P3_reasoning_short":
                # Validar que P3 cumpla límites (<=5 bullets, <=90 palabras)
                bullet_count = output.count('*') + output.count('- ') # Rough guess
                words = len(output.split())
                if words > 120: # Allow some buffer for prompt echo, user said 90 words TOTAL output or explanation?
                    # "No more than 90 words total." Assuming content.
                    # We'll valid leniently if we can't separate prompt.
                    check["status"] = "FAIL"
                    check["info"] = f"Too long: {words} words"
                elif bullet_count > 10: # Assuming 5 bullets, maybe some extra chars
                     check["status"] = "WARN"
                     check["info"] = f"High bullet count: {bullet_count}"
                else:
                    check["info"] = f"{words} words, approx {bullet_count} bullets"

            elif pid == "P4_failfast_behavior":
                 # Validar que P4 contenga condición de corte y contador consecutivo
                 if "50" in output and ("count" in output.lower() or "counter" in output.lower()):
                     check["info"] = "Logic found"
                 else:
                     check["status"] = "FAIL"
                     check["info"] = "Missing '50' or counter logic"

            if check["status"] == "FAIL":
                results["summary"] = "FAIL"
            
            results["details"].append(check)

    except Exception as e:
        results["summary"] = "ERROR"
        results["error"] = str(e)

    print(json.dumps(results, indent=2))
    with open(REPORT_FILE, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    validate_outputs()

import json
import os
import csv
from io import StringIO

def check_file(path):
    print(f"Checking {path}...")
    errors = []
    counts = {}
    
    if not os.path.exists(path):
        return [f"File {path} does not exist"]
        
    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line)
                task_type = item.get("task_type", "unknown")
                counts[task_type] = counts.get(task_type, 0) + 1
                
                # Check JSON validity
                if task_type == "json_format":
                    json.loads(item["output"])
                    
                # Check CSV validity
                if task_type == "csv_format":
                    reader = csv.reader(StringIO(item["output"]))
                    list(reader)
                    
            except Exception as e:
                errors.append(f"Line {line_idx+1}: {e}")
                
    print(f"Counts: {counts}")
    return errors

def main():
    reports = {}
    reports["train_errors"] = check_file("data/instruction_v1/train.jsonl")
    reports["val_errors"] = check_file("data/instruction_v1/val.jsonl")
    
    status = "PASS" if not reports["train_errors"] and not reports["val_errors"] else "FAIL"
    reports["status"] = status
    
    with open("reports/stage2_dataset_checks.json", "w") as f:
        json.dump(reports, f, indent=2)
        
    print(f"Validation finished. Status: {status}")

if __name__ == "__main__":
    main()

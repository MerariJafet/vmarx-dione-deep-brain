import os
import json

CHECKPOINT_PATH = os.getenv("PHASE_A_CHECKPOINT", "models/training_v1_0/checkpoint_final")
REQUIRED_FILES = ["model.safetensors", "optimizer.bin", "random_states_0.pkl"]

def check_inventory():
    inventory = {"exists": False, "files": {}, "missing": []}
    
    if os.path.isdir(CHECKPOINT_PATH):
        inventory["exists"] = True
        for f in os.listdir(CHECKPOINT_PATH):
            fp = os.path.join(CHECKPOINT_PATH, f)
            if os.path.isfile(fp):
                inventory["files"][f] = os.path.getsize(fp)
    
        for req in REQUIRED_FILES:
            if req not in inventory["files"]:
                inventory["missing"].append(req)
    else:
        inventory["error"] = "Directory not found"

    print(json.dumps(inventory, indent=2))
    with open("reports/phaseA_checkpoint_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

if __name__ == "__main__":
    check_inventory()

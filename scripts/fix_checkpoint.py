
import os
import sys
from safetensors.torch import load_file, save_file

def fix_checkpoint(checkpoint_dir):
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading {model_path}...")
    state_dict = load_file(model_path)
    
    # Filter keys: Keep only LoRA parameters
    # LoRA params usually contain "lora_" or "modules_to_save"
    # We explicitly exclude "base_layer" which corresponds to frozen 4-bit weights
    
    new_state_dict = {}
    removed_count = 0
    kept_count = 0
    
    for key, value in state_dict.items():
        if "base_layer" in key:
            removed_count += 1
            continue
        if "lora_" in key or "modules_to_save" in key:
            new_state_dict[key] = value
            kept_count += 1
        else:
            # Dangerous to remove unknown keys, but base_model keys usually don't have lora_
            # If it's pure LoRA, everything we want has lora_
            # Check for other potential keys like bias if trainable?
            # In QLoRA, biases are usually frozen unless specified.
            if "bias" in key and "lora" not in key:
                 # If bias is frozen, we don't save it. If trainable, it should be in modules_to_save?
                 # Safe bet: If not lora, drop it for now as fix for base model weights.
                 pass
            
            # Print a few skipped ones to be sure
            if removed_count < 5:
                print(f"Skipping key: {key}")
            removed_count += 1

    print(f"Kept {kept_count} keys. Removed {removed_count} keys.")
    
    if kept_count == 0:
        print("Error: No LoRA keys found! Aborting save to prevent data loss.")
        return

    print(f"Saving fixed checkpoint to {model_path}...")
    save_file(new_state_dict, model_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pythonFixCheckpoint.py <checkpoint_dir>")
        sys.exit(1)
    
    fix_checkpoint(sys.argv[1])

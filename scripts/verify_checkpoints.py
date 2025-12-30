
import os
import glob
from pathlib import Path

def check_checkpoints():
    pattern = "checkpoints/pilot_v0_1_run_*"
    dirs = glob.glob(pattern)
    if not dirs:
        print("No pilot run directories found.")
        return

    # Sort by creation time (latest last)
    dirs.sort(key=os.path.getctime)
    latest_dir = dirs[-1]
    
    print(f"Latest Run Directory: {latest_dir}")
    
    try:
        checkpoints = sorted([d for d in os.listdir(latest_dir) if d.startswith("checkpoint_")])
        print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        
        # Verify content of last checkpoint
        if checkpoints:
            last_ckpt = os.path.join(latest_dir, checkpoints[-1])
            contents = os.listdir(last_ckpt)
            print(f"Contents of {checkpoints[-1]}: {contents}")
            
            # Check for specific files expected by accelerate/pytorch
            required = ['scheduler.bin', 'optimizer.bin', 'random_states_0.pkl'] 
            # Note: actual filenames vary by accelerate version and config (e.g. optimizer.bin vs pytorch_model.bin)
            # Just print what's there.
    except Exception as e:
        print(f"Error reading directory: {e}")

if __name__ == "__main__":
    check_checkpoints()

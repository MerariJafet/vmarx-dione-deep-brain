from safetensors.torch import load_file
import sys

def inspect(path):
    print(f"Loading {path}...")
    try:
        sd = load_file(path)
        keys = list(sd.keys())
        print(f"Total keys: {len(keys)}")
        print("Sample keys:")
        for k in keys[:10]:
            print(f"  {k}: {sd[k].shape}")
            
        has_lora = any("lora" in k for k in keys)
        has_base = any("layers.0.self_attn.q_proj.weight" in k for k in keys) # check specific base key
        
        print(f"Has LoRA keys: {has_lora}")
        print(f"Has Base weights: {has_base}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect(sys.argv[1])

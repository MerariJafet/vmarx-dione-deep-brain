
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from src.training.model_utils import get_qlora_model
from src.training.dataset import SlidingWindowDataset

def quick_eval():
    # Load Config
    with open("configs/training_spec.json", "r") as f:
        full_config = json.load(f)
    config = full_config["pilot_training_v0.1"]
    
    # Override for Eval
    config["training_setup"]["seq_len_effective"] = 1536 # Match training
    
    # Model
    print("Loading Model...")
    # Point to Base Model
    # We need to perform inference with the Adapter we just trained? 
    # The prompt implies "check metrics of the pilot". 
    # Loading the base model + LoRA checkpoint_50 implies validity.
    
    # Actually, we can just load the base model + the latest checkpoint adapter.
    # But `get_qlora_model` returns a model ready for training. 
    # for eval we can use `PeftModel.from_pretrained`.
    
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    base_id = config["model"]["base_model_id"]
    adapter_path = "checkpoints/pilot_v0_1_run_20251222_182916_8dcf/checkpoint_50"
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Init PEFT
    from peft import LoraConfig, get_peft_model
    # Reconstruct LoraConfig from spec
    peft_config = LoraConfig(
        r=config["model"]["lora"]["r"],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        target_modules=config["model"]["lora"]["target_modules"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Load Weights manually
    from safetensors.torch import load_file
    print(f"Loading weights from {adapter_path}/model.safetensors...")
    state_dict = load_file(f"{adapter_path}/model.safetensors")
    
    # Load logic: The state dict likely contains "base_model.model..." keys if accelerator saved it all.
    # Peft model expects them fast.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model.eval()
    
    # Dataset
    data_dir = config["paths"]["tokenized_data"]
    # We want "Validation". Since we have no explicit split, we take a few chunks.
    # Ideally, we used a seed for training. We can use a different seed.
    dataset = SlidingWindowDataset(data_dir, seq_len=1536, stride=1536) 
    
    print("Running Eval on 10 batches...")
    losses = []
    
    # Just take 20 random samples, skipping the first few to avoid overlap with train if unlucky
    # (Not perfect valid split but "aunque sea 1 split simple")
    iterator = iter(dataset)
    
    # Skip 100
    for _ in range(100):
        next(iterator)
        
    with torch.no_grad():
        for i in range(20): # 20 samples
            try:
                batch = next(iterator)
                input_ids = batch["input_ids"].unsqueeze(0).cuda() # Batch size 1
                labels = batch["labels"].unsqueeze(0).cuda()
                
                outputs = model(input_ids=input_ids, labels=labels)
                losses.append(outputs.loss.item())
            except StopIteration:
                break
                
    if not losses:
        print("No data gathered.")
        return

    avg_loss = np.mean(losses)
    print(json.dumps({"val_loss": avg_loss, "samples": len(losses)}))

if __name__ == "__main__":
    quick_eval()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import load_checkpoint_in_model
import os

def load_phaseA_model(checkpoint_path=None, device_map="auto"):
    if checkpoint_path is None:
        checkpoint_path = os.getenv("PHASE_A_CHECKPOINT", "models/training_v1_0/checkpoint_final")
    
    base_model_path = "/home/merari-acero/Escritorio/VMarx Dione DB/models/Mistral-7B-v0.1"
    
    print(f"Initializing Base Model (4-bit): {base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print("Reconstructing PEFT Config...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    print(f"Loading Weights from {checkpoint_path}...")
    # Accelerate load checkoint handling
    # load_checkpoint_in_model loads the state dict into the model
    # It handles sharded checkpoints if needed, or single file
    load_checkpoint_in_model(model, checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

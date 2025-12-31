import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import logging
import os

logger = logging.getLogger("training.model")

def load_model_weights_only(model, checkpoint_path):
    import os
    from safetensors.torch import load_file
    
    # accelerator saves as model.safetensors or pytorch_model.bin
    sf_path = os.path.join(checkpoint_path, "model.safetensors")
    pt_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    state_dict = None
    if os.path.exists(sf_path):
        logger.info(f"Loading weights from {sf_path}...")
        state_dict = load_file(sf_path, device="cpu")
    elif os.path.exists(pt_path):
        logger.info(f"Loading weights from {pt_path}...")
        state_dict = torch.load(pt_path, map_location="cpu")
    else:
        logger.warning(f"No weight file found in {checkpoint_path}")
        return
        
    # We need to handle the case where the state_dict keys match the unwrapped model
    # Accelerator saves the 'prepared' model, which might have 'module.' or 'base_model.' prefixes.
    # PeftModel has 'base_model.model...'
    # Let's try loading with strict=False and reporting.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if len(unexpected) > 400:
        # Heuristic: if many unexpected, maybe we need to wrap the model or skip prefixes?
        # Standard PEFT load usually handles this, but here we are doing it manually.
        pass

def get_qlora_model(config, checkpoint_path=None):
    model_id = config["paths"].get("base_model", "mistralai/Mistral-7B-v0.1")
    
    logger.info(f"Loading model {model_id} in 4-bit...")
    # ...
    
    compute_dtype = torch.float32
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    kwargs = {
        "quantization_config": bnb_config,
        "device_map": {"": 0},
        "trust_remote_code": True
    }
    
    if os.path.isdir(model_id):
        kwargs["local_files_only"] = True
        
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    
    lora_cfg = config["training_setup"].get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    if checkpoint_path:
        load_model_weights_only(model, checkpoint_path)
    
    print("\n[T3 Check] Trainable Parameters:")
    model.print_trainable_parameters()
    print("--------------------------------------------------\n")
    
    return model

def get_tokenizer(config):
    model_id = config["paths"].get("base_model", "mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

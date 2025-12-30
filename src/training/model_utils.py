import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import logging

logger = logging.getLogger("training.model")

def get_qlora_model(config):
    model_id = config["model"].get("base_model_id", "mistralai/Mistral-7B-v0.1")
    
    logger.info(f"Loading model {model_id} in 4-bit...")
    
    # Use float32 for compute_dtype to ensure maximum stability with BF16 training
    compute_dtype = torch.float32
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    # Check if local path
    import os
    kwargs = {
        "quantization_config": bnb_config,
        "device_map": {"": 0},
        "trust_remote_code": True
    }
    
    if os.path.isdir(model_id):
        logger.info(f"Loading model from local directory: {model_id}")
        kwargs["local_files_only"] = True
    else:
        logger.info(f"Loading model {model_id} from Hub (4-bit)...")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
        
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)
    
    # LoRA
    lora_cfg = config["model"].get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    # T3 Check: Confirm 4-bit and LoRA
    print("\n[T3 Check] Model Structure (Layer 0):")
    print(model.model.model.layers[0])
    print("[T3 Check] Trainable Parameters:")
    model.print_trainable_parameters()
    print("--------------------------------------------------\n")
    
    return model

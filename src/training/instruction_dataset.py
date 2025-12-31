import torch
from torch.utils.data import Dataset
import json
import logging

logger = logging.getLogger("training.instruction_dataset")

class InstructionDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_length=1024):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        logger.info(f"Loading instruction dataset from {jsonl_path}...")
        with open(jsonl_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        logger.info(f"Loaded {len(self.examples)} examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        
        instruction = item.get("instruction", "")
        input_data = item.get("input", "")
        response = item.get("output", "")
        
        # Template
        prompt = f"Instruction:\n{instruction}\n"
        if input_data:
            prompt += f"\nInput:\n{input_data}\n"
        prompt += f"\nResponse:\n"
        
        full_text = prompt + response + self.tokenizer.eos_token
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        
        # Create Labels (masking the prompt)
        # We only want to calculate loss on the 'Response' part.
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None
        )
        prompt_len = len(prompt_encodings["input_ids"])
        
        # Labels: same as input_ids but with -100 for the prompt part
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Pad manually if needed (actually DataLoader/Collate should handle this)
        # But we yield tensors here.
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "metadata": {
                "id": item.get("id", "unknown"),
                "task_type": item.get("task_type", "unknown")
            }
        }

def instruction_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) # assumed pad_token_id=0 or tokenizer.pad_token_id
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": padded_attention_mask
    }

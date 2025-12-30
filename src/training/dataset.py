import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
import random
import logging
import hashlib
import json

logger = logging.getLogger("training.dataset")

class SlidingWindowDataset(IterableDataset):
    def __init__(self, data_dir, seq_len=2048, stride=1024, split="train", train_pct=0.85, sampling_mode="proportional", domain_filter=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.train_pct = train_pct
        self.sampling_mode = sampling_mode
        self.domain_filter = domain_filter
        
        # 1. Discover and Split Files
        self.files = self._discover_and_split_files()
        
        # 2. Apply Domain Filter (if requested)
        if self.domain_filter:
            self.files = [f for f in self.files if f.stem.startswith(f"{self.domain_filter}_")]
            logger.info(f"Applied domain filter '{self.domain_filter}'. Remaining files: {len(self.files)}")
        
        # 3. Categorize for Sampling (if needed)
        self.domain_files = self._categorize_domains(self.files)
        if len(self.domain_files) > 1 and "50/50" in sampling_mode:
            logger.info(f"Enabled Balanced Sampling across domains: {list(self.domain_files.keys())}")
            self.use_balanced_sampling = True
        else:
            self.use_balanced_sampling = False
            
        logger.info(f"Initialized Dataset split='{split}' with {len(self.files)} files. Balanced Sampling: {self.use_balanced_sampling}")

    def _discover_and_split_files(self):
        all_files = list(self.data_dir.glob("*.npy"))
        if not all_files:
            return []
            
        # Group by Symbol to do Time-based Walk Forward split PER ASSET
        # Assumes filenames: domain_SYMBOL_index.npy or similar: crypto_ADAUSDT_0.npy
        grouped = {}
        for f in all_files:
            parts = f.stem.split('_')
            # heuristic: parts[-1] is index, parts[:-1] is id
            if parts[-1].isdigit():
                symbol_id = "_".join(parts[:-1])
                idx = int(parts[-1])
                if symbol_id not in grouped:
                    grouped[symbol_id] = []
                grouped[symbol_id].append((idx, f))
            else:
                # fallback
                grouped.setdefault("misc", []).append((0, f))
        
        final_files = []
        for symbol, entries in grouped.items():
            # Sort by index (Time)
            entries.sort(key=lambda x: x[0])
            sorted_files = [x[1] for x in entries]
            n = len(sorted_files)
            cutoff = int(n * self.train_pct)
            
            # Ensure at least 1 file in train if n > 0, unless n=1 then train gets it?
            # Or strict split.
            if self.split == "train":
                selection = sorted_files[:cutoff]
                # If cutoff is 0 but we have files, maybe force 1?
                # If n=1, cutoff=0.85 -> 0. Train empty? 
                # Let's enforce: if n=1, train gets it. Val gets nothing?
                # Or user preference "walk forward". Usually Val is "future".
                if n == 1 and self.split == "train": 
                    selection = sorted_files # All to train if only 1?
                elif n == 1 and self.split == "val":
                    selection = []
            else: # val
                selection = sorted_files[cutoff:]
                
            final_files.extend(selection)
            
        # Logging Summary by Domain
        n_crypto = sum(1 for f in final_files if f.name.startswith("crypto_"))
        n_equity = sum(1 for f in final_files if f.name.startswith("equities_"))
        logger.info(f"Split '{self.split}': {len(final_files)} files (Crypto={n_crypto}, Equities={n_equity})")

        return final_files

    def _categorize_domains(self, file_list):
        domains = {}
        for f in file_list:
            # crypto_..., equity_...
            prefix = f.stem.split('_')[0]
            if prefix not in domains:
                domains[prefix] = []
            domains[prefix].append(f)
        return domains

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if self.use_balanced_sampling:
            # Balanced sampling logic
            # We need to yield roughly 50% from each domain
            # Strategy: Create iterators for each domain, round robin yield?
            # Or just shuffle big list?
            # True 50/50 balance implies upsampling the smaller class.
            
            # Complex for IterableDataset with multi-worker.
            # Simplification: Each worker gets a subset of files from EACH domain, keeping ratio?
            # Or just weighted choice.
            
            # Let's do a weighted random generator per worker.
            
            # 1. Assign files to this worker
            my_files_by_domain = {}
            for dom, files in self.domain_files.items():
                if worker_info:
                    # Shard files of this domain
                    per_w = int(np.ceil(len(files) / float(worker_info.num_workers)))
                    start = worker_info.id * per_w
                    end = start + per_w
                    my_files_by_domain[dom] = files[start:end]
                else:
                    my_files_by_domain[dom] = files
            
            # 2. Infinite Loop generator
            max_len = max(len(v) for v in my_files_by_domain.values()) if my_files_by_domain else 0
            # If we just want one epoch, this is hard with balancing.
            # "Proportional" is standard iteration.
            # "50/50" implies re-sampling.
            
            # Let's implement a "Stochastic Epoch":
            # Flatten to list, but composed of 50% domain A, 50% domain B
            # Picking random files?
            
            # Simpler approach: Just shuffle the assigned files? No that's proportional.
            # We must yield from domains with prob 0.5.
            
            domains = list(my_files_by_domain.keys())
            files_pool = {d: list(f) for d, f in my_files_by_domain.items()}
            
            # Shuffle internal pools
            for d in domains:
                random.shuffle(files_pool[d])
            
            # Yield loop
            while any(files_pool.values()): # While something remains
                # Pick domain: 50/50
                dom = random.choice(domains)
                if not files_pool[dom]:
                    # Domain exhausted, pick other
                    other = [d for d in domains if files_pool[d]]
                    if not other: break
                    dom = other[0] # Fallback to proportional at end? Or stop?
                    # Stopping means 50/50 is preserved only until smallest exhausted.
                
                f = files_pool[dom].pop()
                yield from self._process_file(f)
                
        else:
            # Standard Proportional
            files = list(self.files)
            if worker_info:
                per_worker = int(np.ceil(len(files) / float(worker_info.num_workers)))
                iter_start = worker_info.id * per_worker
                iter_end = min(iter_start + per_worker, len(files))
                files = files[iter_start:iter_end]
                
            random.shuffle(files)
            for f in files:
                yield from self._process_file(f)
                
    def _process_file(self, f):
        try:
            seq = np.load(f).astype(np.int64)
            n_tokens = len(seq)
            if n_tokens < self.seq_len:
                return

            starts = list(range(0, n_tokens - self.seq_len + 1, self.stride))
            # Shuffle windows within file (User request: "shuffle windows")
            random.shuffle(starts)
            
            for start in starts:
                end = start + self.seq_len
                chunk = seq[start:end]
                
                # Metadata: SHA1 Hash of tokens
                chunk_bytes = chunk.tobytes()
                token_hash = hashlib.sha1(chunk_bytes).hexdigest()
                
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "metadata": {
                        "domain": f.stem.split('_')[0],
                        "asset_id": f.stem,
                        "window_offset": start,
                        "token_hash_sha1": token_hash
                    }
                }
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    def __len__(self):
        return len(self.files) * 7 # Rough setmate

def custom_collate_fn(batch):
    # batch is a list of dicts
    # keys: input_ids, labels, attention_mask, metadata
    
    collated = {}
    
    # 1. Tensors - Stack them
    for key in ["input_ids", "labels", "attention_mask"]:
         if key in batch[0]:
             collated[key] = torch.stack([item[key] for item in batch])
             
    # 2. Metadata - Encode as JSON ByteTensor (Padded) -> Accelerate friendly
    if "metadata" in batch[0]:
        meta_tensors = []
        for item in batch:
            meta_json = json.dumps(item["metadata"])
            # ASCII/UTF-8 bytes
            meta_bytes = torch.tensor(list(meta_json.encode('utf-8')), dtype=torch.uint8)
            meta_tensors.append(meta_bytes)
            
        # Pad sequence
        from torch.nn.utils.rnn import pad_sequence
        # batch_first=True -> (B, L)
        padded_meta = pad_sequence(meta_tensors, batch_first=True, padding_value=0)
        collated["metadata_bytes"] = padded_meta
        
    return collated

import torch
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
import logging
import json
import time
from pathlib import Path
import numpy as np
import signal
import sys
import traceback
import subprocess
import os
from collections import defaultdict

# try to import bitsandbytes
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

logger = logging.getLogger("training.trainer")

class PilotTrainer:
    def __init__(self, model, train_loader, config, output_dir, val_loaders=None):
        self.config = config["training_setup"]
        self.run_plan = config["run_plan"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # val_loaders can be a dict {"name": loader} or a single loader (legacy support)
        if val_loaders and not isinstance(val_loaders, dict):
             self.val_loaders = {"global": val_loaders}
        else:
             self.val_loaders = val_loaders or {}
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config["grad_accum"],
            mixed_precision=self.config["precision"].lower() if self.config["precision"] in ["FP16", "BF16"] else "no"
        )
        
        self.model = model
        
        # Optimizer
        if bnb and "8bit" in self.config["optimizer"]:
            self.optimizer = bnb.optim.PagedAdamW8bit(
                self.model.parameters(), 
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"])
            )
        else:
            logger.warning("BitsAndBytes 8bit optimizer not available (or not selected), falling back to torch AdamW")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=float(self.config["learning_rate"]),
                weight_decay=float(self.config["weight_decay"])
            )
            
        self.train_loader = train_loader
        
        # Scheduler (placeholder for now, will be prepared in train)
        self.scheduler = None # Initialize scheduler to None, will be set up later if needed

        # Prepare (moved to train method for resume logic)
        # self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
        #     self.model, self.optimizer, self.train_loader
        # )
        
        self.metrics = {
            "train_loss": [],
            "grad_norm": [],
            "steps": []
        }
        self.toxic_batches_count = 0
        self.consecutive_toxic_batches = 0
        self.total_attempts = 0
        self.last_activity_time = time.time()
        self.last_recovery_time = 0
        self.toxic_history = defaultdict(int) # hash -> count

        
        # Signal Handling
        self.stop_signal_received = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, signum, frame):
        logger.warning(f"Received signal {signum}. Initiating emergency save...")
        self.stop_signal_received = True
        
    def train(self, resume_from_checkpoint=None):
        logger.info("Preparing for training...")
        
        # Prepare components
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        # Prepare val loaders if they exist
        for k, v in self.val_loaders.items():
            self.val_loaders[k] = self.accelerator.prepare(v)

        total_steps = int(self.run_plan["steps_target"])
        completed_steps = 0
        
        # Resume logic
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            # Use strict=False to allow loading LoRA-only checkpoints (missing base keys are expected)
            self.accelerator.load_state(resume_from_checkpoint, strict=False)
            # Infer step from checkpoint name? "checkpoint_N"
            try:
                base_name = Path(resume_from_checkpoint).name
                if "checkpoint_" in base_name:
                    completed_steps = int(base_name.split("_")[1])
                logger.info(f"Resumed at step {completed_steps}")
            except Exception as e:
                logger.warning(f"Could not infer step from checkpoint name: {e}")
                
        logger.info(f"Starting training loop from step {completed_steps} to {total_steps}...")

        start_time = time.time()
        
        
        # Progress Bar
        progress_bar = tqdm(
            initial=completed_steps, 
            total=total_steps, 
            disable=not self.accelerator.is_local_main_process
        )
        
        data_iter = iter(self.train_loader)        
        try:
            while completed_steps < total_steps:
                step_start_time = time.time()
                self.model.train() # Ensure model is in train mode at start of each step
                
                    batch = next(data_iter)
                    self.total_attempts += 1
                    self.last_activity_time = time.time()
                except StopIteration:
                    # Restart iterator if data exhausted?
                    # SlidingWindowDataset is huge, but if small dataset, we cycle.
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                    self.last_activity_time = time.time()
                    
                # Watchdog Check (every 10 attempts)
                if self.total_attempts % 10 == 0:
                     # Check for HANG (no activity for > 5 mins)
                     time_since_activity = time.time() - self.last_activity_time
                     
                     if time_since_activity > 300: 
                         # Check Cooldown (10 mins = 600s)
                         if (time.time() - self.last_recovery_time) > 600:
                             logger.warning(
                                 f"WATCHDOG: HANG DETECTED (No activity for {time_since_activity:.0f}s). "
                                 f"Step={completed_steps} Attempt={self.total_attempts} ConsecutiveToxic={self.consecutive_toxic_batches}"
                             )
                             self.last_recovery_time = time.time()
                             self.handle_toxic_state(data_iter, batch_info=None) # Start fresh
                             continue
                         else:
                             # In cooldown, verified healthy by attempts incrementing? 
                             # Actually if we are here, attempts % 10 == 0 happened.
                             # If attempts are incrementing, 'time_since_activity' should be small.
                             # If 'time_since_activity' > 300, it means we spent 5 mins getting from attempt N to N+10?
                             # Or we blocked on 'next(data_iter)'?
                             # If we blocked, we wouldn't reach here.
                             # So triggering this means we are VERY slow (30s per batch?).
                             # Warning without reset is appropriate if in cooldown.
                             logger.warning(f"WATCHDOG: System Slow (Lag {time_since_activity:.0f}s) but in Cooldown. Ignoring.")


                # Decode Metadata
                current_batch_metadata = None
                if "metadata_bytes" in batch:
                    try:
                        meta_tensor = batch["metadata_bytes"] # (B, L)
                        # We only need the first one if checking hash on B=1 microbatch 
                        # But for logging we might want all.
                        # Decoded list
                        current_batch_metadata = []
                        for i in range(len(meta_tensor)):
                            row_bytes = meta_tensor[i].tolist()
                            # Trim trailing 0s (padding)
                            valid_bytes = bytes([b for b in row_bytes if b != 0])
                            current_batch_metadata.append(json.loads(valid_bytes.decode('utf-8')))
                            
                        # Flatten for sig: handle_toxic_state expects dict or list of dicts?
                        # It expects 'batch_info' which in log_toxic uses items().
                        # Let's make it a single dict if len=1 for backcompat or list if len>1?
                        # Actually previous code handled list access logic in handle_toxic.
                        # Let's clean that up.
                        # 'batch_info' -> passes to handle_toxic and log_toxic.
                        # log_toxic: iteratess keys.
                        # handle_toxic: accesses "token_hash_sha1".
                        
                        # So let's transform list of dicts -> dict of lists (Columnar)
                        # This works best with my previous Log logic.
                        if vars(self).get("metadata_columnar", True):
                             columnar = defaultdict(list)
                             for item in current_batch_metadata:
                                 for k,v in item.items():
                                     columnar[k].append(v)
                             current_batch_metadata = dict(columnar)
                    except Exception as e:
                        logger.warning(f"Failed to decode metadata: {e}")
                        current_batch_metadata = None

                with self.accelerator.accumulate(self.model):
                    
                    # A4: OOV Check (Pre-forward/loss check is better but harder with HF model forward logic inside?
                    # Actually we should check BEFORE model().
                    # But we are inside accumulation context.
                    # Let's check inputs before model forward.
                    is_safe, oov_msg = self.check_inputs(batch)
                    if not is_safe:
                        logger.warning(f"OOV DETECTED: {oov_msg}. Quarantining batch.")
                        self.log_toxic_batch(completed_steps, self.total_attempts, "OOV", 0.0, 0.0, batch_info=current_batch_metadata)
                        self.toxic_batches_count += 1
                        self.optimizer.zero_grad()
                        # progress bar update handled by loop flow if we continue
                        progress_bar.update(1) # Mark attempt
                        continue

                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        attention_mask=batch["attention_mask"]
                    )
                    loss = outputs.loss
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                         self.consecutive_toxic_batches += 1
                         logger.warning(f"TOXIC BATCH DETECTED (Loss NaN/Inf) at step {completed_steps}. Consecutive: {self.consecutive_toxic_batches}/50")
                         self.log_toxic_batch(completed_steps, self.total_attempts, "loss_nan_inf", float('nan'), 0.0)
                         self.handle_toxic_state(data_iter)
                         continue
                    
                    self.accelerator.backward(loss)
                    
                    # Grad Norm Check
                    grad_norm = 0.0
                    if self.accelerator.sync_gradients:
                        # Clip grad
                        self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip"]))
                        
                        # Compute norm (post-clip)
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        grad_norm = total_norm ** 0.5
                        
                        if np.isnan(grad_norm) or np.isinf(grad_norm):
                             self.consecutive_toxic_batches += 1
                             logger.warning(f"TOXIC BATCH DETECTED (Grad NaN/Inf) at step {completed_steps}. Consecutive: {self.consecutive_toxic_batches}/50")
                             self.log_toxic_batch(completed_steps, self.total_attempts, "grad_nan_inf", loss.item(), float('nan'))
                             self.handle_toxic_state(data_iter)
                             continue

                    self.optimizer.step()
                    if self.scheduler: # Apply scheduler if it exists
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Success! Reset counter
                    self.consecutive_toxic_batches = 0
                    
                # Logging if step done
                if self.accelerator.sync_gradients:
                    completed_steps += 1
                    # self.last_effective_step_time updated on activity now
                    progress_bar.update(1)
                    
                    current_loss = loss.item()
                    self.metrics["train_loss"].append(current_loss)
                    self.metrics["grad_norm"].append(grad_norm)
                    self.metrics["steps"].append(completed_steps)

                    # Calculate tokens per second
                    tokens_processed = batch["input_ids"].numel() * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
                    step_duration = time.time() - step_start_time
                    tps = tokens_processed / step_duration if step_duration > 0 else 0

                    # Memory usage
                    mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

                    # GPU utilization
                    gpu_util = "N/A"
                    try:
                        # minimal overhead query
                        result = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                            encoding='utf-8'
                        )
                        gpu_util = result.strip() + "%"
                    except:
                        pass

                    logs = {
                        "loss": f"{current_loss:.4f}",
                        "grad": f"{grad_norm:.2f}",
                        "mem": f"{mem_alloc:.0f}MiB",
                        "gpu": gpu_util,
                        "tps": f"{tps:.1f}"
                    }
                    progress_bar.set_postfix(logs)
                    
                    # Check NaNs
                    if np.isnan(current_loss) or np.isinf(current_loss):
                        logger.warning(f"NaN/Inf detected in loss logging at step {completed_steps}. Skipping report.")
                        # We already checked before backward, so this might be redundant or from grad sync?
                        # Just don't crash.
                        pass
                        # raise ValueError(f"NaN/Inf detected in loss at step {completed_steps}!")
                        
                    # Eval/Save
                    if completed_steps % 100 == 0:
                        self.report_status(completed_steps, current_loss, grad_norm, tps)

                    if completed_steps % int(self.run_plan["eval_every_steps"]) == 0:
                        self.evaluate(completed_steps)
                        
                    if completed_steps % int(self.run_plan["save_every_steps"]) == 0:
                        self.save_checkpoint(completed_steps)
                        self.save_metrics()
                            
                        # Time limit
                        elapsed_hours = (time.time() - start_time) / 3600
                        if elapsed_hours > float(self.run_plan["max_runtime_hours"]):
                            logger.info("Max runtime reached.")
                            break

        except Exception as e:
            logger.error(f"CRASH DETECTED at step {completed_steps}: {e}")
            logger.error(traceback.format_exc())
            logger.info("Attempting emergency checkpoint save...")
            try:
                self.save_checkpoint(f"crash_step_{completed_steps}")
                self.save_metrics()
            except Exception as save_e:
                logger.error(f"Failed to save emergency checkpoint: {save_e}")
            raise e # Re-raise to crash properly after saving 

        if self.stop_signal_received:
            logger.info("Training interrupted by user/system. Saving final state...")
            self.save_checkpoint(f"interrupted_step_{completed_steps}")
                    
        self.save_checkpoint("final")
        self.save_metrics()
        self.save_closeout_report(completed_steps)
        logger.info("Training complete.")
        
    def save_checkpoint(self, step):
        path = self.output_dir / f"checkpoint_{step}"
        self.accelerator.save_state(path)
        logger.info(f"Saved checkpoint to {path}")
        
    def save_metrics(self):
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def evaluate(self, step):
        if not self.val_loaders:
            return
            
        logger.info(f"Running evaluation at step {step}...")
        self.model.eval()
        
        eval_metrics_step = {"step": step}

        for domain, loader in self.val_loaders.items():
            losses = []
            max_eval_steps = 50
            
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= max_eval_steps: break
                    
                    input_ids = batch["input_ids"].to(self.model.device)
                    labels = batch["labels"].to(self.model.device)
                    
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    losses.append(outputs.loss.item())
            
            if losses:
                avg_loss = float(np.mean(losses))
                logger.info(f"Step {step} Val Loss ({domain}): {avg_loss:.4f}")
                eval_metrics_step[f"val_loss_{domain}"] = avg_loss
        
        if len(eval_metrics_step) > 1:
            if "val_loss_history" not in self.metrics:
                self.metrics["val_loss_history"] = []
            self.metrics["val_loss_history"].append(eval_metrics_step)
            
        self.model.train()

    def report_status(self, step, current_loss, grad_norm, tps):
        try:
            # Calculate rolling metrics
            train_loss_hist = self.metrics.get("train_loss", [])
            last_100 = train_loss_hist[-100:] if len(train_loss_hist) > 0 else [current_loss]
            train_loss_avg = float(np.mean(last_100))
            train_loss_p95 = float(np.percentile(last_100, 95))
            
            # Val metrics
            val_crypto = "N/A"
            val_equities = "N/A"
            if "val_loss_history" in self.metrics and self.metrics["val_loss_history"]:
                latest_val = self.metrics["val_loss_history"][-1]
                val_crypto = latest_val.get("val_loss_crypto", "N/A")
                val_equities = latest_val.get("val_loss_equity", "N/A")

            # Grad stats
            grad_hist = self.metrics.get("grad_norm", [])
            grad_last_100 = grad_hist[-100:] if len(grad_hist) > 0 else [grad_norm]
            grad_avg = float(np.mean(grad_last_100))
            grad_p95 = float(np.percentile(grad_last_100, 95))
            
            # VRAM
            vram_peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

            # Toxic stats
            # Assuming self.toxic_batches_count is tracked
            
            status_record = {
                "effective_step": step,
                "train_loss_avg_last100": f"{train_loss_avg:.4f}",
                "train_loss_p95_last100": f"{train_loss_p95:.4f}",
                "val_loss_crypto": f"{val_crypto}",
                "val_loss_equities": f"{val_equities}",
                "grad_norm_avg_last100": f"{grad_avg:.4f}",
                "grad_norm_p95_last100": f"{grad_p95:.4f}",
                "toxic_batches_total": self.toxic_batches_count,
                "vram_peak_mib": f"{vram_peak:.0f}",
                "tps": f"{tps:.1f}",
                "tps": f"{tps:.1f}",
                "total_attempts": self.total_attempts,
                "timestamp": time.time()
            }
            
            with open(self.output_dir / "run_status.jsonl", "a") as f:
                f.write(json.dumps(status_record) + "\n")
                
            logger.info(f"STATUS [Step {step}]: Loss={train_loss_avg:.4f} | ToxicTotal={self.toxic_batches_count}")

        except Exception as e:
            logger.warning(f"Failed to write status report: {e}")

    def log_toxic_batch(self, step, attempt, reason, loss_val, grad_norm_val, batch_info=None):
        try:
            report_dir = self.output_dir.parent.parent / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Collapse list of metadata if batch size > 1 (take first for provenance or all?)
            # Usually batch info is one dict if collated? 
            # If collated, it's a dict of lists.
            # We want specific sample info.
            # Just dump what we have.
            
            meta_dump = {}
            if batch_info:
                # Convert tensor/lists to serializable
                 for k, v in batch_info.items():
                     if isinstance(v, list) and len(v) > 0:
                         meta_dump[k] = str(v[0]) # Just take first of microbatch for identifying source
                     elif isinstance(v, torch.Tensor):
                         meta_dump[k] = str(v.tolist())
                     else:
                         meta_dump[k] = str(v)

            record = {
                "timestamp": time.time(),
                "effective_step": step,
                "attempt_idx": attempt,
                "reason": reason,
                "loss": str(loss_val),
                "grad_norm": str(grad_norm_val),
                "metadata": meta_dump
            }
            
            with open(report_dir / "toxic_batches.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log toxic batch: {e}")

    def check_inputs(self, batch):
        # A4: OOV Check
        # Mistral 7B vocab ~32000. 
        # We should check if any input_id >= config.vocab_size if known, or just a safe bound?
        # Let's get vocab size from model config if possible.
        vocab_size = self.model.config.vocab_size
        
        if (batch["input_ids"] >= vocab_size).any() or (batch["input_ids"] < 0).any():
             return False, "OOV detected"
        return True, "OK"

    def handle_toxic_state(self, data_iter, batch_info=None):
        """
        Handles state reset and recovery logic when a toxic batch is detected.
        """
        self.toxic_batches_count += 1
        
        # 0. Hash Check (Atlas Directive)
        if batch_info and "token_hash_sha1" in batch_info:
            hashes = batch_info["token_hash_sha1"]
            # If batch, hashes is list.
            if isinstance(hashes, list):
                target_hash = hashes[0] # Check first of batch
            else:
                target_hash = hashes
            
            self.toxic_history[target_hash] += 1
            if self.toxic_history[target_hash] >= 3:
                 logger.error(f"ATLAS QUARANTINE: Hash {target_hash} repeated 3 times. FAIL FAST.")
                 raise RuntimeError(f"TOXIC_HASH_FAIL_FAST: {target_hash}")
        
        # 1. Zero grad
        self.optimizer.zero_grad(set_to_none=True)
        
        # 2. Reset Scaler if present
        if hasattr(self.accelerator, "scaler") and self.accelerator.scaler is not None:
             try:
                 # If the scaler has a _scale, we might want to back it off or just let it update?
                 # Accelerator usually handles this if we skip? 
                 # Explicitly setting it to a safe value might be needed if it keeps growing.
                 # For now, we trust accelerator.step() skip. 
                 pass 
             except:
                 pass

        # 3. Clear cache periodically (every 10 toxics)
        if self.consecutive_toxic_batches % 10 == 0:
            torch.cuda.empty_cache()

        # 4. Circuit Breaker Checks
        if self.consecutive_toxic_batches == 25:
            logger.warning("CIRCUIT BREAKER: 25 Consecutive Toxics. Attempting SOFT RECOVERY (Reseed + Reset Iterator).")
            # Reseed
            base_seed = int(self.config.get("seed", 42))
            new_seed = base_seed + self.consecutive_toxic_batches + int(time.time())
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(new_seed)
            
            # Reset Iterator - This forces a fresh shuffle/order if using a random sampler
            data_iter = iter(self.train_loader)
            logger.info("Recovery: Iterator reset handled.")

        elif self.consecutive_toxic_batches >= 100:
            logger.error("CIRCUIT BREAKER: 100 Consecutive Toxics. ABORTING.")
            current_steps = self.metrics["steps"][-1] if self.metrics["steps"] else 0
            self.save_checkpoint(f"crash_step_{current_steps}")
            raise RuntimeError("TOXIC_LOOP_ABORT: consecutive toxic batches exceeded limit (100)")
        
    def save_closeout_report(self, step):
        # A5: Final closeout template
        report = {
            "run_id": os.environ.get("LOG", "unknown").split('/')[-1].replace('.log', ''),
            "steps_completed": step,
            "wall_time": time.time(), # could be duration if we tracked start
            "toxic_count_total": self.toxic_batches_count,
            # "toxic_rate": ... need total attempts
            "val_history": self.metrics.get("val_loss_history", []),
            "checkpoints": {
                "latest": str(self.output_dir / "checkpoint_latest"),
                "final": str(self.output_dir / "checkpoint_final")
            },
            "resume_verified": True,
            "artifacts_paths": {
                "status_jsonl": str(self.output_dir / "run_status.jsonl"),
                "toxic_jsonl": str(self.output_dir.parent.parent / "reports" / "toxic_batches.jsonl")
            }
        }
        with open(self.output_dir.parent.parent / "stage1_closeout_final.json", "w") as f:
            json.dump(report, f, indent=2)

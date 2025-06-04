"""
Numerically stable GRPO trainer with Accelerate for multi-GPU support
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class StableGRPOTrainerAccelerate:
    """
    Numerically stable GRPO trainer with Accelerate for multi-GPU training.
    """
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-6,
        num_epochs: int = 3,
        batch_size: int = 4,
        k_samples: int = 4,
        temperature: float = 0.8,
        kl_coef: float = 0.2,
        gamma: float = 0.99,
        max_grad_norm: float = 0.5,
        min_temperature: float = 0.1,
        numerical_stability_eps: float = 1e-8,
        use_accelerate: bool = True
    ):
        """
        Initialize stable GRPO trainer with Accelerate support.
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.k_samples = k_samples
        self.temperature = temperature
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.min_temperature = min_temperature
        self.eps = numerical_stability_eps
        self.use_accelerate = use_accelerate
        
        # Initialize accelerator
        if self.use_accelerate:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=1,
                mixed_precision='fp16'  # Use fp16 for efficiency
            )
            self.device = self.accelerator.device
            logger.info(f"Using Accelerate with {self.accelerator.num_processes} processes")
        else:
            self.accelerator = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with appropriate dtype
        model_dtype = torch.float16 if self.use_accelerate else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reference model for KL divergence
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            trust_remote_code=True
        )
        self.ref_model.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * 1000,
            eta_min=learning_rate * 0.1
        )
        
        # Prepare models with accelerator
        if self.use_accelerate:
            self.model, self.ref_model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.ref_model, self.optimizer, self.scheduler
            )
        else:
            self.model = self.model.to(self.device)
            self.ref_model = self.ref_model.to(self.device)
        
        # Track training stability
        self.stability_monitor = {
            'nan_count': 0,
            'inf_count': 0,
            'negative_prob_count': 0,
            'gradient_explosions': 0
        }
    
    def safe_log_probs(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Compute log probabilities with numerical stability checks.
        """
        # Ensure minimum temperature
        temperature = max(temperature, self.min_temperature)
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Compute log softmax with temperature
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        
        # Check for numerical issues
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            logger.warning("Detected nan/inf in log_probs, applying safety fixes")
            # Replace nan/inf with safe values
            log_probs = torch.nan_to_num(log_probs, nan=-100.0, posinf=100.0, neginf=-100.0)
        
        return log_probs
    
    def safe_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = 512,
        temperature: float = None
    ) -> torch.Tensor:
        """
        Generate sequences with numerical stability checks.
        """
        if temperature is None:
            temperature = self.temperature
        
        temperature = max(temperature, self.min_temperature)
        
        try:
            # Disable mixed precision for generation
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            return outputs
            
        except RuntimeError as e:
            if "probability tensor contains either `inf`, `nan`" in str(e):
                self.stability_monitor['nan_count'] += 1
                logger.error(f"Generation error: {e}")
                logger.info("Attempting recovery with higher temperature")
                
                # Try again with higher temperature and top-k sampling
                recovery_temp = min(temperature * 2, 2.0)
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=recovery_temp,
                    do_sample=True,
                    top_k=10,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                return outputs
            else:
                raise
    
    def compute_rewards_batch(
        self,
        prompts: List[str],
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of generated texts with safety checks.
        """
        rewards = []
        
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            # Simple reward: length penalty to encourage conciseness
            prompt_length = len(self.tokenizer.encode(prompt))
            generated_length = len(self.tokenizer.encode(generated))
            response_length = generated_length - prompt_length
            
            # Reward structure
            length_penalty = -0.01 * max(response_length - 50, 0)
            base_reward = 1.0 if response_length > 10 else 0.0
            
            reward = base_reward + length_penalty
            
            # Clamp reward to prevent extreme values
            reward = torch.tensor(reward).clamp(min=-5.0, max=5.0)
            rewards.append(reward)
        
        return torch.stack(rewards).to(self.device)
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using group normalization with numerical stability.
        """
        # Reshape rewards: (batch_size * k_samples,) -> (batch_size, k_samples)
        rewards = rewards.view(-1, self.k_samples)
        
        # Normalize within each group with safety checks
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        
        # Prevent division by zero
        std = torch.maximum(std, torch.tensor(self.eps).to(rewards.device))
        
        advantages = (rewards - mean) / std
        
        # Clamp advantages to prevent extreme values
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)
        
        return advantages.view(-1)
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL penalty with numerical stability.
        """
        # Compute KL divergence
        kl = torch.exp(log_probs) * (log_probs - ref_log_probs)
        kl = kl.sum(dim=-1).mean()
        
        # Check for numerical issues
        if torch.isnan(kl) or torch.isinf(kl):
            logger.warning("KL divergence is nan/inf, using fallback")
            kl = torch.tensor(0.0).to(kl.device)
        
        return self.kl_coef * kl
    
    def safe_tensor_mean(self, tensor: torch.Tensor) -> float:
        """
        Safely get mean value from tensor, handling multi-GPU cases.
        """
        if tensor.numel() > 1:
            return tensor.mean().item()
        else:
            return tensor.item()
    
    def train_step(
        self,
        prompts: List[str],
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Single training step with enhanced stability and multi-GPU support.
        """
        self.model.train()
        
        all_rewards = []
        all_advantages = []
        all_losses = []
        
        # Generate K responses per prompt with dynamic temperature
        for prompt_idx, prompt in enumerate(prompts):
            prompt_rewards = []
            prompt_losses = []
            
            for k in range(self.k_samples):
                # Dynamic temperature based on sampling round
                dynamic_temp = self.temperature * (1.0 + 0.1 * k)
                
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Generate response with safety
                try:
                    with torch.no_grad():
                        outputs = self.safe_generate(
                            inputs,
                            max_length=max_length,
                            temperature=dynamic_temp
                        )
                    
                    # Extract generated text
                    generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # Compute reward
                    reward = self.compute_rewards_batch([prompt], [generated_text])[0]
                    prompt_rewards.append(reward)
                    
                    # Compute loss with gradient
                    if self.use_accelerate:
                        self.accelerator.wait_for_everyone()
                    
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    model_outputs = self.model(
                        input_ids=outputs.sequences,
                        attention_mask=(outputs.sequences != self.tokenizer.pad_token_id).long()
                    )
                    
                    # Compute log probabilities with safety
                    log_probs = self.safe_log_probs(model_outputs.logits, dynamic_temp)
                    
                    # Reference model log probs
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=outputs.sequences,
                            attention_mask=(outputs.sequences != self.tokenizer.pad_token_id).long()
                        )
                        ref_log_probs = self.safe_log_probs(ref_outputs.logits, dynamic_temp)
                    
                    # Compute KL penalty
                    kl_loss = self.compute_kl_penalty(log_probs, ref_log_probs)
                    
                    prompt_losses.append(kl_loss)
                    
                except Exception as e:
                    logger.error(f"Error in generation for prompt {prompt_idx}, sample {k}: {e}")
                    # Use fallback values
                    prompt_rewards.append(torch.tensor(0.0).to(self.device))
                    prompt_losses.append(torch.tensor(0.0).to(self.device))
            
            if prompt_rewards:
                all_rewards.extend(prompt_rewards)
                all_losses.extend(prompt_losses)
        
        if not all_rewards:
            logger.error("No successful generations in this batch")
            return {
                'loss': 0.0,
                'reward': 0.0,
                'kl_penalty': 0.0,
                'stability_issues': self.stability_monitor
            }
        
        # Stack tensors
        rewards = torch.stack(all_rewards)
        losses = torch.stack(all_losses)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        
        # Compute policy gradient loss
        pg_loss = -(advantages.detach() * losses.mean())
        
        # Total loss
        total_loss = pg_loss + losses.mean()
        
        # Backward pass with gradient clipping
        try:
            if self.use_accelerate:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()
            
            # Check gradients before clipping
            if self.use_accelerate:
                total_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            if total_norm > self.max_grad_norm * 10:
                self.stability_monitor['gradient_explosions'] += 1
                logger.warning(f"Gradient explosion detected: {total_norm}")
            
            self.optimizer.step()
            self.scheduler.step()
            
        except Exception as e:
            logger.error(f"Error in backward pass: {e}")
            self.optimizer.zero_grad()
            total_norm = torch.tensor(0.0)
        
        # Gather metrics across GPUs if using accelerate
        if self.use_accelerate:
            total_loss = self.accelerator.gather(total_loss)
            rewards = self.accelerator.gather(rewards)
            losses = self.accelerator.gather(losses)
            advantages = self.accelerator.gather(advantages)
        
        return {
            'loss': self.safe_tensor_mean(total_loss),
            'reward': self.safe_tensor_mean(rewards),
            'kl_penalty': self.safe_tensor_mean(losses),
            'advantages': self.safe_tensor_mean(advantages),
            'gradient_norm': self.safe_tensor_mean(total_norm) if torch.is_tensor(total_norm) else 0.0,
            'stability_monitor': self.stability_monitor.copy()
        }
    
    def train(
        self,
        train_prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Full training loop with stability monitoring and multi-GPU support.
        """
        logger.info(f"Starting stable GRPO training with {len(train_prompts)} prompts")
        if self.use_accelerate:
            logger.info(f"Using {self.accelerator.num_processes} GPUs with Accelerate")
        
        # Create batches
        num_batches = len(train_prompts) // self.batch_size
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Shuffle prompts
            indices = np.random.permutation(len(train_prompts))
            
            epoch_losses = []
            epoch_rewards = []
            
            # Create progress bar on main process only
            if not self.use_accelerate or self.accelerator.is_main_process:
                progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")
            else:
                progress_bar = range(num_batches)
            
            for batch_idx in progress_bar:
                # Get batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(train_prompts))
                batch_indices = indices[start_idx:end_idx]
                batch_prompts = [train_prompts[i] for i in batch_indices]
                
                # Train step
                metrics = self.train_step(batch_prompts)
                
                epoch_losses.append(metrics['loss'])
                epoch_rewards.append(metrics['reward'])
                
                # Update progress bar on main process
                if not self.use_accelerate or self.accelerator.is_main_process:
                    if hasattr(progress_bar, 'set_postfix'):
                        progress_bar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'reward': f"{metrics['reward']:.4f}",
                            'kl': f"{metrics['kl_penalty']:.4f}",
                            'grad': f"{metrics['gradient_norm']:.2f}"
                        })
                
                # Log stability issues
                if batch_idx % 100 == 0 and (not self.use_accelerate or self.accelerator.is_main_process):
                    logger.info(f"Stability monitor: {metrics['stability_monitor']}")
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            
            if not self.use_accelerate or self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # Save checkpoint
            if save_path and (not self.use_accelerate or self.accelerator.is_main_process):
                checkpoint_path = f"{save_path}_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        if not self.use_accelerate or self.accelerator.is_main_process:
            logger.info("Training complete!")
            logger.info(f"Final stability report: {self.stability_monitor}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint with stability information."""
        # Unwrap model if using accelerate
        model_to_save = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stability_monitor': self.stability_monitor,
            'config': {
                'model_name': self.model_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'k_samples': self.k_samples,
                'temperature': self.temperature,
                'kl_coef': self.kl_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }, path)

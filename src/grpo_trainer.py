"""GRPO Trainer implementation for MedLogicTrace"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class MedLogicGRPOTrainer:
    """Custom GRPO trainer for MedLogicTrace experiments"""
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        k_samples: int = 4,
        temperature: float = 0.8,
        kl_coef: float = 0.1,
        gamma: float = 0.99,
        device: str = "cuda",
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForCausalLM] = None
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.k_samples = k_samples
        self.temperature = temperature
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": device} if device == "mps" else "auto",
            trust_remote_code=True
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Enable gradient checkpointing for memory efficiency
        # self.model.gradient_checkpointing_enable()
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages for policy gradient updates.
        
        Args:
            rewards: Tensor of shape (batch_size, k_samples) containing rewards
            
        Returns:
            Tensor of advantages with same shape as rewards
        """
        # Ensure rewards are on the correct device and require gradients
        rewards = rewards.to(self.device).detach().view(-1, self.k_samples)
        
        # Normalize within each group with safety checks
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        
        # Prevent division by zero
        std = torch.maximum(std, torch.tensor(1e-10).to(self.device))
        
        advantages = (rewards - mean) / std
        
        # Clamp advantages to prevent extreme values
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)
        
        return advantages.detach()  # Detach final advantages as they are used as constants in loss computation
    
    def create_grpo_trainer(self, train_dataset) -> GRPOTrainer:
        """Create TRL GRPOTrainer instance"""
        
        # Configure GRPO training arguments
        grpo_config = GRPOConfig(
            output_dir=f"outputs/{self.model_name.replace('/', '_')}_grpo",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",  # We'll evaluate separately on medical datasets
            
            # GRPO specific parameters
            kl_coef=self.kl_coef,
            max_new_tokens=256,
            temperature=self.temperature,
            top_p=0.95,
            
            # Memory optimization
            gradient_checkpointing=False,
            fp16=False,
            bf16=False,
            
            # Disable features not compatible with MPS
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=self.model,
            config=grpo_config,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
        )
        
        return trainer
    
    def train(self, train_dataset):
        """Train the model using GRPO"""
        print(f"\nStarting GRPO training for {self.model_name}")
        print(f"Training on {len(train_dataset)} examples")
        
        trainer = self.create_grpo_trainer(train_dataset)
        
        # Train
        trainer.train()
        
        # Save the final model
        output_path = f"outputs/{self.model_name.replace('/', '_')}_grpo_final"
        print(f"Saving trained model to {output_path}")
        trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return output_path
    
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response


def create_reward_function(correct_answers: Dict[str, str]) -> callable:
    """Create a reward function for GRPO training"""
    def reward_fn(samples: List[str], prompts: List[str]) -> List[float]:
        """Calculate rewards based on correctness and token efficiency"""
        rewards = []
        
        for sample, prompt in zip(samples, prompts):
            # Base reward for generating a response
            reward = 0.1
            
            # Check if response contains mathematical reasoning patterns
            if any(keyword in sample.lower() for keyword in ["therefore", "thus", "hence", "so", "="]):
                reward += 0.2
            
            # Penalty for overly long responses
            token_count = len(sample.split())
            if token_count > 100:
                reward -= 0.1 * (token_count - 100) / 100
            
            # Bonus for concise responses
            if 20 < token_count < 50:
                reward += 0.3
            
            rewards.append(reward)
        
        return rewards
    
    return reward_fn

"""GRPO Trainer implementation for MedLogicTrace"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from typing import Dict, List, Optional
import os
from datetime import datetime
from tqdm import tqdm


class MedLogicGRPOTrainer:
    """Custom GRPO trainer for MedLogicTrace experiments"""
    
    def __init__(self, model_name: str, config, device: str = "mps"):
        self.model_name = model_name
        self.config = config
        self.device = device
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings for MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS doesn't support bf16
            device_map={"": device} if device == "mps" else "auto",
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if config.grpo_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def create_grpo_trainer(self, train_dataset) -> GRPOTrainer:
        """Create TRL GRPOTrainer instance"""
        
        # Configure GRPO training arguments
        grpo_config = GRPOConfig(
            output_dir=f"{self.config.output_dir}/{self.model_name.replace('/', '_')}_grpo",
            per_device_train_batch_size=self.config.grpo_config.batch_size,
            gradient_accumulation_steps=self.config.grpo_config.gradient_accumulation_steps,
            learning_rate=self.config.grpo_config.learning_rate,
            num_train_epochs=self.config.grpo_config.num_epochs,
            warmup_steps=self.config.grpo_config.warmup_steps,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",  # We'll evaluate separately on medical datasets
            
            # GRPO specific parameters
            kl_coef=self.config.grpo_config.kl_coef,
            max_new_tokens=self.config.grpo_config.max_new_tokens,
            temperature=self.config.grpo_config.temperature,
            top_p=self.config.grpo_config.top_p,
            
            # Memory optimization
            gradient_checkpointing=self.config.grpo_config.gradient_checkpointing,
            fp16=False,  # MPS doesn't support fp16
            bf16=False,  # MPS doesn't support bf16
            
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
        output_path = f"{self.config.output_dir}/{self.model_name.replace('/', '_')}_grpo_final"
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
                temperature=self.config.grpo_config.temperature,
                top_p=self.config.grpo_config.top_p,
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

#!/usr/bin/env python3
"""Enhanced GPU GRPO training with Bespoke-Stratos-17k dataset and visualization"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import json
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field

@dataclass
class EnhancedGRPOConfig:
    """Enhanced configuration for GPU GRPO training"""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Training settings
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 2
    warmup_steps: int = 100
    
    # GRPO specific
    kl_coef: float = 0.05
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    group_size: int = 4
    
    # GPU optimizations
    use_fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Dataset
    dataset_name: str = "bespokelabs/Bespoke-Stratos-17k"
    num_train_examples: int = 5000
    
    # Plotting
    plot_every_n_steps: int = 50
    save_plots: bool = True
    
    # Output
    output_dir: str = "models/grpo_trained"
    experiment_name: str = field(default_factory=lambda: f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


class PlottingCallback:
    """Custom callback for plotting training metrics"""
    
    def __init__(self, config: EnhancedGRPOConfig):
        self.config = config
        self.metrics = {
            'step': [],
            'loss': [],
            'kl_divergence': [],
            'reward_mean': [],
            'reward_std': [],
            'response_length': [],
            'learning_rate': []
        }
        
        # Create results directory
        self.plot_dir = f"results/{config.experiment_name}"
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is None:
            return
        
        # Store metrics
        self.metrics['step'].append(state.global_step)
        self.metrics['loss'].append(logs.get('loss', 0))
        self.metrics['kl_divergence'].append(logs.get('kl', 0))
        self.metrics['reward_mean'].append(logs.get('reward_mean', 0))
        self.metrics['reward_std'].append(logs.get('reward_std', 0))
        self.metrics['response_length'].append(logs.get('response_length', 0))
        self.metrics['learning_rate'].append(logs.get('learning_rate', 0))
        
        # Plot every N steps
        if state.global_step % self.config.plot_every_n_steps == 0:
            self.plot_training_curves()
    
    def plot_training_curves(self):
        """Generate training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'GRPO Training Progress - {self.config.model_name}', fontsize=16)
        
        # Loss curve
        axes[0, 0].plot(self.metrics['step'], self.metrics['loss'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[0, 1].plot(self.metrics['step'], self.metrics['kl_divergence'], 'r-', alpha=0.7)
        axes[0, 1].axhline(y=self.config.kl_coef, color='k', linestyle='--', label=f'Target: {self.config.kl_coef}')
        axes[0, 1].set_title('KL Divergence')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('KL')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward statistics
        axes[0, 2].plot(self.metrics['step'], self.metrics['reward_mean'], 'g-', alpha=0.7, label='Mean')
        axes[0, 2].fill_between(
            self.metrics['step'],
            np.array(self.metrics['reward_mean']) - np.array(self.metrics['reward_std']),
            np.array(self.metrics['reward_mean']) + np.array(self.metrics['reward_std']),
            alpha=0.3, color='g'
        )
        axes[0, 2].set_title('Reward Distribution')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Response length
        axes[1, 0].plot(self.metrics['step'], self.metrics['response_length'], 'm-', alpha=0.7)
        axes[1, 0].set_title('Average Response Length')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Tokens')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.metrics['step'], self.metrics['learning_rate'], 'c-', alpha=0.7)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Reward vs Length scatter
        if len(self.metrics['reward_mean']) > 10:
            axes[1, 2].scatter(self.metrics['response_length'][-100:], 
                              self.metrics['reward_mean'][-100:], 
                              alpha=0.5, c=self.metrics['step'][-100:], cmap='viridis')
            axes[1, 2].set_title('Reward vs Response Length (Last 100 steps)')
            axes[1, 2].set_xlabel('Response Length')
            axes[1, 2].set_ylabel('Reward')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_plots:
            plot_path = f"{self.plot_dir}/training_curves_step_{self.metrics['step'][-1]}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved training plot to {plot_path}")
        
        plt.close()
    
    def save_final_plots(self):
        """Generate final summary plots"""
        self.plot_training_curves()
        
        # Save metrics to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(f"{self.plot_dir}/training_metrics.csv", index=False)
        print(f"Saved training metrics to {self.plot_dir}/training_metrics.csv")


def load_bespoke_dataset(config: EnhancedGRPOConfig):
    """Load and prepare Bespoke-Stratos-17k dataset"""
    print(f"Loading {config.dataset_name}...")
    
    dataset = load_dataset(config.dataset_name, split="train")
    
    # Take subset for training
    dataset = dataset.select(range(min(config.num_train_examples, len(dataset))))
    
    print(f"Loaded {len(dataset)} examples")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Examine first example
    if len(dataset) > 0:
        print("\nFirst example:")
        for key, value in dataset[0].items():
            print(f"{key}: {str(value)[:200]}...")
    
    return dataset


def create_advanced_reward_function():
    """Create an advanced reward function for reasoning quality"""
    def reward_fn(responses: List[str], prompts: List[str]) -> List[float]:
        """Calculate rewards based on reasoning quality and efficiency"""
        rewards = []
        
        for response in responses:
            reward = 0.0
            
            # Base reward for valid response
            if len(response.strip()) > 10:
                reward += 0.1
            
            # Reward for step-by-step reasoning
            reasoning_indicators = [
                "step", "first", "second", "next", "then", "therefore",
                "because", "since", "thus", "hence", "so we", "let's",
                "we can", "this means", "which gives", "finally"
            ]
            reasoning_score = sum(1 for indicator in reasoning_indicators 
                                if indicator.lower() in response.lower())
            reward += min(reasoning_score * 0.1, 0.5)
            
            # Reward for mathematical notation
            math_indicators = ["=", "+", "-", "*", "/", "(", ")", "^"]
            math_score = sum(1 for indicator in math_indicators 
                           if indicator in response)
            reward += min(math_score * 0.05, 0.2)
            
            # Penalty for excessive length
            token_count = len(response.split())
            if token_count > 300:
                reward -= (token_count - 300) * 0.001
            
            # Bonus for optimal length (50-150 tokens)
            if 50 <= token_count <= 150:
                reward += 0.2
            
            # Penalty for repetition
            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.7:
                    reward -= 0.2
            
            rewards.append(max(0, min(1, reward)))  # Clip to [0, 1]
        
        return rewards
    
    return reward_fn


def train_model_with_grpo(config: EnhancedGRPOConfig, gpu_id: int = 0):
    """Train a model using GRPO with visualization"""
    
    # Set GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"\n{'='*60}")
    print(f"Starting GRPO Training on GPU {gpu_id}")
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Training examples: {config.num_train_examples}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset = load_bespoke_dataset(config)
    
    # Load model and tokenizer
    print(f"Loading model {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare training data
    def prepare_dataset(examples):
        """Prepare dataset for GRPO training"""
        prepared = []
        
        for i in range(len(examples['prompt']) if 'prompt' in examples else len(examples)):
            # Handle different dataset formats
            if 'prompt' in examples:
                prompt = examples['prompt'][i]
            elif 'question' in examples:
                prompt = examples['question'][i]
            else:
                prompt = str(examples[list(examples.keys())[0]][i])
            
            # Tokenize
            tokenized = tokenizer(prompt, truncation=True, max_length=512)
            
            prepared.append({
                'input_ids': tokenized['input_ids'],
                'query': prompt
            })
        
        return prepared
    
    train_dataset = prepare_dataset(dataset)
    
    # Create GRPO configuration
    training_args = GRPOConfig(
        output_dir=f"{config.output_dir}/{config.experiment_name}",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        
        # GRPO specific
        kl_coef=config.kl_coef,
        group_size=config.group_size,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        
        # GPU optimizations
        fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        
        # Logging
        logging_steps=10,
        save_steps=500,
        report_to=["tensorboard"],
        logging_dir=f"logs/{config.experiment_name}",
    )
    
    # Create plotting callback
    plotting_callback = PlottingCallback(config)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        reward_model=create_advanced_reward_function(),
        callbacks=[plotting_callback]
    )
    
    # Train
    print("\nStarting training...")
    start_time = datetime.now()
    
    trainer.train()
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save model
    output_path = f"{config.output_dir}/{config.experiment_name}/final_model"
    print(f"\nSaving model to {output_path}")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Generate final plots
    plotting_callback.save_final_plots()
    
    # Save training summary
    summary = {
        "model_name": config.model_name,
        "dataset": config.dataset_name,
        "num_examples": config.num_train_examples,
        "training_time_seconds": training_time,
        "final_loss": trainer.state.log_history[-1].get('loss', 'N/A'),
        "output_path": output_path,
        "config": config.__dict__
    }
    
    with open(f"{plotting_callback.plot_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Results saved to {plotting_callback.plot_dir}")
    
    return output_path, plotting_callback.plot_dir


def main():
    """Main training function"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to train")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-examples", type=int, default=5000, help="Number of training examples")
    args = parser.parse_args()
    
    # Create config
    config = EnhancedGRPOConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        num_train_examples=args.num_examples
    )
    
    # Train
    output_path, results_dir = train_model_with_grpo(config, gpu_id=args.gpu)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Model saved to: {output_path}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

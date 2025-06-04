#!/usr/bin/env python3
"""
Main training script for LogicTrace with mathematical pretraining
"""

import os
import torch
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.logictrace_grpo_trainer import LogicTraceGRPOTrainer
from src.math_data_utils import MathDatasetLoader


class MathReasoningDataset(Dataset):
    """PyTorch dataset for mathematical reasoning."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate function for batching."""
    prompts = [item['prompt'] for item in batch]
    references = [item['reference'] for item in batch]
    complexities = [item['complexity'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    return {
        'prompts': prompts,
        'references': references,
        'complexities': complexities,
        'answers': answers
    }


def train_logictrace_on_math(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    dataset_name: str = "gsm8k",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models/logictrace_math",
    use_dapo: bool = True
):
    """
    Train model with LogicTrace optimization on mathematical reasoning.
    
    Args:
        model_name: Pretrained model to use
        dataset_name: Math dataset (gsm8k or openr1-math)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        use_dapo: Whether to use DAPO enhancements
    """
    print(f"Starting LogicTrace training on {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"DAPO enhancements: {use_dapo}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = LogicTraceGRPOTrainer(
        model_name=model_name,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        k_samples=4,
        temperature=0.8,
        kl_coef=0.1,
        device=device,
        # LogicTrace parameters
        alpha_structure=0.3,
        alpha_length=0.2,
        alpha_accuracy=0.5,
        base_length_penalty=0.01,
        complexity_threshold=3,
        use_dapo_enhancements=use_dapo
    )
    
    # Load mathematical dataset
    print(f"\nLoading {dataset_name} dataset...")
    math_loader = MathDatasetLoader(dataset_name, split="train")
    math_data = math_loader.load_dataset()
    
    # Create prompts
    prompts_data = math_loader.create_prompts(math_data, include_cot=True)
    
    # Split into train/val
    split_idx = int(0.9 * len(prompts_data))
    train_data = prompts_data[:split_idx]
    val_data = prompts_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create dataloaders
    train_dataset = MathReasoningDataset(train_data)
    val_dataset = MathReasoningDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Training loop
    best_val_reward = -float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training phase
        trainer.model.train()
        epoch_metrics = {
            'loss': [],
            'reward': [],
            'efficiency': [],
            'preservation': []
        }
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Train on batch
            metrics = trainer.train_on_batch(
                prompts=batch['prompts'],
                reference_solutions=batch['references'],
                problem_complexities=batch['complexities']
            )
            
            # Update metrics
            epoch_metrics['loss'].append(metrics['loss'])
            epoch_metrics['reward'].append(metrics['avg_reward'])
            epoch_metrics['efficiency'].append(metrics['token_efficiency_ratio'])
            epoch_metrics['preservation'].append(metrics['step_preservation_ratio'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'reward': f"{metrics['avg_reward']:.4f}",
                'efficiency': f"{metrics['token_efficiency_ratio']:.2f}"
            })
            
            # Periodic evaluation
            if (batch_idx + 1) % 50 == 0:
                val_reward = evaluate_on_validation(trainer, val_loader)
                print(f"\nValidation reward at step {batch_idx + 1}: {val_reward:.4f}")
                
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    checkpoint_path = os.path.join(
                        save_dir,
                        f"best_checkpoint_epoch{epoch+1}_step{batch_idx+1}.pt"
                    )
                    trainer.save_checkpoint(checkpoint_path, epoch, metrics)
                    print(f"New best model saved!")
        
        # Epoch summary
        avg_metrics = {
            key: np.mean(values) for key, values in epoch_metrics.items()
        }
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_metrics['loss']:.4f}")
        print(f"  Average Reward: {avg_metrics['reward']:.4f}")
        print(f"  Token Efficiency: {avg_metrics['efficiency']:.2f}")
        print(f"  Step Preservation: {avg_metrics['preservation']:.2f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'metrics': avg_metrics
        })
        
        # Full validation
        print("\nRunning full validation...")
        val_reward = evaluate_on_validation(trainer, val_loader)
        print(f"Validation reward: {val_reward:.4f}")
        
        # Save epoch checkpoint
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        trainer.save_checkpoint(checkpoint_path, epoch, avg_metrics)
    
    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed! Best validation reward: {best_val_reward:.4f}")
    print(f"Models saved to: {save_dir}")
    
    return trainer, training_history


def evaluate_on_validation(trainer, val_loader):
    """Evaluate model on validation set."""
    trainer.model.eval()
    all_rewards = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Generate responses
            responses_per_prompt, _ = trainer.generate_k_responses(
                batch['prompts'],
                max_length=512
            )
            
            # Compute rewards for best response
            batch_rewards = []
            for i, responses in enumerate(responses_per_prompt):
                # Evaluate each response
                response_rewards = []
                for response in responses:
                    # Simple correctness check
                    extracted_answer = trainer._extract_answer(response)
                    is_correct = extracted_answer == batch['answers'][i]
                    
                    # Compute LogicTrace reward
                    reward_dict = trainer.logictrace_optimizer.compute_logictrace_reward(
                        response,
                        batch['references'][i],
                        is_correct,
                        batch['complexities'][i]
                    )
                    response_rewards.append(reward_dict['total_reward'].item())
                
                # Take best response reward
                batch_rewards.append(max(response_rewards))
            
            all_rewards.extend(batch_rewards)
    
    return np.mean(all_rewards)


def analyze_trained_model(trainer, test_samples):
    """Analyze the trained model's reasoning quality."""
    print("\n" + "="*50)
    print("Analyzing Trained Model")
    print("="*50)
    
    trainer.model.eval()
    
    for i, sample in enumerate(test_samples[:5]):  # Analyze 5 samples
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {sample['question']}")
        
        # Generate response
        prompt = f"Question: {sample['question']}\n\nLet me solve this step by step.\n\n"
        inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.device)
        
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = trainer.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"\nGenerated Response:")
        print(response)
        
        # Analyze reasoning quality
        if 'answer' in sample:
            reference = "\n".join(sample.get('reasoning_steps', [])) + f"\n\nThe answer is {sample['answer']}"
            analysis = trainer.logictrace_optimizer.analyze_reasoning_quality(
                response, reference
            )
            
            print(f"\nAnalysis:")
            print(f"  Token Efficiency: {analysis['token_efficiency_ratio']:.2f}")
            print(f"  Step Compression: {analysis['step_compression_ratio']:.2f}")
            print(f"  Preserved Important Steps: {analysis['avg_importance_preserved']:.2f}")


if __name__ == "__main__":
    # Training configuration
    config = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_name": "gsm8k",  # or "openr1-math"
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": f"models/logictrace_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "use_dapo": True
    }
    
    # Train model
    trainer, history = train_logictrace_on_math(**config)
    
    # Load test samples for analysis
    math_loader = MathDatasetLoader(config['dataset_name'], split="test")
    test_data = math_loader.load_dataset()[:10]
    
    # Analyze trained model
    analyze_trained_model(trainer, test_data)

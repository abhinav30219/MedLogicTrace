#!/usr/bin/env python3
"""
Medical transfer learning script - transfers mathematical reasoning to medical domain
"""

import os
import torch
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.logictrace_grpo_trainer import LogicTraceGRPOTrainer
from src.math_data_utils import MathToMedicalTransferDataset
from src.data_utils import load_medical_datasets
from src.medical_evaluator import MedicalEvaluator


class MedicalTransferDataset(Dataset):
    """PyTorch dataset for medical transfer learning."""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_medical_batch(batch):
    """Custom collate function for medical batches."""
    prompts = [item['prompt'] for item in batch]
    references = [item.get('reference', '') for item in batch]
    complexities = [item.get('complexity', 2) for item in batch]
    stages = [item.get('stage', 'clinical_reasoning') for item in batch]
    
    return {
        'prompts': prompts,
        'references': references,
        'complexities': complexities,
        'stages': stages
    }


def run_medical_transfer(
    pretrained_checkpoint: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    medical_datasets: list = ["medmcqa", "pubmedqa"],
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-6,  # Lower LR for fine-tuning
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "models/logictrace_medical",
    progressive_transfer: bool = True
):
    """
    Transfer mathematical reasoning capabilities to medical domain.
    
    Args:
        pretrained_checkpoint: Path to pretrained math model
        model_name: Base model name
        medical_datasets: List of medical datasets to use
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (lower for fine-tuning)
        device: Device to train on
        save_dir: Directory to save models
        progressive_transfer: Whether to use curriculum learning
    """
    print(f"Starting Medical Transfer Learning")
    print(f"Pretrained checkpoint: {pretrained_checkpoint}")
    print(f"Medical datasets: {medical_datasets}")
    print(f"Progressive transfer: {progressive_transfer}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize trainer with pretrained model
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
        alpha_structure=0.35,  # Slightly higher for medical
        alpha_length=0.15,     # Lower penalty for medical explanations
        alpha_accuracy=0.50,
        base_length_penalty=0.005,  # Lighter penalty
        complexity_threshold=4,     # Medical problems are more complex
        use_dapo_enhancements=True
    )
    
    # Load pretrained checkpoint
    if os.path.exists(pretrained_checkpoint):
        print(f"Loading pretrained weights from {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained weights loaded successfully!")
    
    # Create transfer curriculum
    transfer_dataset = MathToMedicalTransferDataset()
    
    if progressive_transfer:
        # Progressive curriculum learning
        curriculum = transfer_dataset.create_transfer_curriculum()
        
        for stage_idx, (stage_name, stage_data) in enumerate(curriculum.items()):
            print(f"\n{'='*50}")
            print(f"Stage {stage_idx + 1}: {stage_name}")
            print(f"{'='*50}")
            
            # Create dataloader for this stage
            stage_dataset = MedicalTransferDataset(stage_data)
            stage_loader = DataLoader(
                stage_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_medical_batch
            )
            
            # Train on this stage
            train_stage(trainer, stage_loader, stage_name, epoch=0)
            
            # Save stage checkpoint
            stage_checkpoint = os.path.join(
                save_dir, f"stage_{stage_name}_checkpoint.pt"
            )
            trainer.save_checkpoint(stage_checkpoint, stage_idx, {})
    
    # Load actual medical datasets
    print(f"\n{'='*50}")
    print("Fine-tuning on Medical Datasets")
    print(f"{'='*50}")
    
    # Load medical data
    medical_data = load_medical_data(medical_datasets)
    
    # Split into train/val
    split_idx = int(0.9 * len(medical_data))
    train_data = medical_data[:split_idx]
    val_data = medical_data[split_idx:]
    
    print(f"Medical training samples: {len(train_data)}")
    print(f"Medical validation samples: {len(val_data)}")
    
    # Create dataloaders
    train_dataset = MedicalTransferDataset(train_data)
    val_dataset = MedicalTransferDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_medical_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_medical_batch
    )
    
    # Medical fine-tuning
    best_medical_accuracy = 0.0
    training_history = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Medical Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        trainer.model.train()
        epoch_metrics = train_epoch(trainer, train_loader, epoch)
        
        # Validate
        val_metrics = evaluate_medical_performance(trainer, val_loader)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Training Loss: {epoch_metrics['avg_loss']:.4f}")
        print(f"  Training Reward: {epoch_metrics['avg_reward']:.4f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Validation Token Efficiency: {val_metrics['avg_tokens']:.1f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_medical_accuracy:
            best_medical_accuracy = val_metrics['accuracy']
            best_path = os.path.join(save_dir, "best_medical_model.pt")
            trainer.save_checkpoint(best_path, epoch, val_metrics)
            print(f"New best model saved! Accuracy: {best_medical_accuracy:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_metrics': epoch_metrics,
            'val_metrics': val_metrics
        })
    
    # Save training history
    history_path = os.path.join(save_dir, "medical_transfer_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nMedical transfer learning completed!")
    print(f"Best validation accuracy: {best_medical_accuracy:.4f}")
    print(f"Models saved to: {save_dir}")
    
    return trainer, training_history


def train_stage(trainer, stage_loader, stage_name, epoch):
    """Train on a specific curriculum stage."""
    stage_metrics = {
        'loss': [],
        'reward': [],
        'efficiency': []
    }
    
    pbar = tqdm(stage_loader, desc=f"Training {stage_name}")
    for batch in pbar:
        metrics = trainer.train_on_batch(
            prompts=batch['prompts'],
            reference_solutions=batch['references'],
            problem_complexities=batch['complexities']
        )
        
        stage_metrics['loss'].append(metrics['loss'])
        stage_metrics['reward'].append(metrics['avg_reward'])
        stage_metrics['efficiency'].append(metrics['token_efficiency_ratio'])
        
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'reward': f"{metrics['avg_reward']:.4f}"
        })
    
    avg_loss = np.mean(stage_metrics['loss'])
    avg_reward = np.mean(stage_metrics['reward'])
    print(f"\n{stage_name} completed - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")


def train_epoch(trainer, train_loader, epoch):
    """Train for one epoch on medical data."""
    epoch_metrics = {
        'loss': [],
        'reward': [],
        'efficiency': [],
        'preservation': []
    }
    
    pbar = tqdm(train_loader, desc=f"Medical Training Epoch {epoch + 1}")
    for batch in pbar:
        metrics = trainer.train_on_batch(
            prompts=batch['prompts'],
            reference_solutions=batch['references'],
            problem_complexities=batch['complexities']
        )
        
        epoch_metrics['loss'].append(metrics['loss'])
        epoch_metrics['reward'].append(metrics['avg_reward'])
        epoch_metrics['efficiency'].append(metrics['token_efficiency_ratio'])
        epoch_metrics['preservation'].append(metrics['step_preservation_ratio'])
        
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'reward': f"{metrics['avg_reward']:.4f}",
            'eff': f"{metrics['token_efficiency_ratio']:.2f}"
        })
    
    return {
        'avg_loss': np.mean(epoch_metrics['loss']),
        'avg_reward': np.mean(epoch_metrics['reward']),
        'avg_efficiency': np.mean(epoch_metrics['efficiency']),
        'avg_preservation': np.mean(epoch_metrics['preservation'])
    }


def evaluate_medical_performance(trainer, val_loader):
    """Evaluate on medical validation set."""
    trainer.model.eval()
    evaluator = MedicalEvaluator(trainer.tokenizer)
    
    all_results = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Medical Validation"):
            # Generate responses
            for i, prompt in enumerate(batch['prompts']):
                inputs = trainer.tokenizer(prompt, return_tensors="pt").to(trainer.device)
                
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
                
                # Extract question from prompt
                question = prompt.split('\n')[0].replace('Question: ', '')
                
                # Evaluate
                result = evaluator.evaluate_response(question, response)
                result['reference'] = batch['references'][i]
                all_results.append(result)
    
    # Aggregate metrics
    accuracy = np.mean([r['is_correct'] for r in all_results])
    avg_tokens = np.mean([r['num_tokens'] for r in all_results])
    efficiency_score = accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
    
    return {
        'accuracy': accuracy,
        'avg_tokens': avg_tokens,
        'efficiency_score': efficiency_score,
        'num_samples': len(all_results)
    }


def load_medical_data(dataset_names):
    """Load and format medical datasets."""
    all_data = []
    
    for dataset_name in dataset_names:
        if dataset_name == "medmcqa":
            # Load MedMCQA
            data = load_medical_datasets(['medmcqa'], max_samples=1000)
            for item in data:
                prompt = f"Question: {item['question']}\nOptions:\n"
                for opt, text in item['options'].items():
                    prompt += f"{opt}) {text}\n"
                prompt += "\nAnalyze this medical question step by step and provide the correct answer.\n\n"
                
                all_data.append({
                    'prompt': prompt,
                    'reference': f"The correct answer is {item['answer']}",
                    'complexity': 3,
                    'dataset': 'medmcqa'
                })
                
        elif dataset_name == "pubmedqa":
            # Load PubMedQA
            data = load_medical_datasets(['pubmedqa'], max_samples=1000)
            for item in data:
                prompt = f"Context: {item['context'][:500]}...\n\n"
                prompt += f"Question: {item['question']}\n\n"
                prompt += "Based on the context, analyze this question step by step.\n\n"
                
                all_data.append({
                    'prompt': prompt,
                    'reference': f"The answer is {item['answer']}",
                    'complexity': 4,
                    'dataset': 'pubmedqa'
                })
    
    return all_data


def analyze_medical_transfer_results(trainer, test_samples):
    """Analyze the quality of medical reasoning after transfer."""
    print("\n" + "="*50)
    print("Analyzing Medical Transfer Results")
    print("="*50)
    
    trainer.model.eval()
    
    for i, sample in enumerate(test_samples[:5]):
        print(f"\n--- Medical Sample {i+1} ---")
        print(f"Dataset: {sample['dataset']}")
        print(f"Question: {sample['prompt'].split('\\n')[0]}")
        
        # Generate response
        inputs = trainer.tokenizer(sample['prompt'], return_tensors="pt").to(trainer.device)
        
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
        
        print(f"\nGenerated Medical Reasoning:")
        print(response)
        
        # Analyze efficiency
        if sample.get('reference'):
            analysis = trainer.logictrace_optimizer.analyze_reasoning_quality(
                response, sample['reference']
            )
            
            print(f"\nEfficiency Analysis:")
            print(f"  Token Count: {analysis['token_efficiency_ratio']:.2f}x reference")
            print(f"  Reasoning Steps: {analysis['step_compression_ratio']:.2f}x")


if __name__ == "__main__":
    # Configuration
    config = {
        "pretrained_checkpoint": "models/logictrace_math_20250603_130000/best_checkpoint_epoch2_step100.pt",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "medical_datasets": ["medmcqa", "pubmedqa"],
        "num_epochs": 2,
        "batch_size": 4,
        "learning_rate": 5e-6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": f"models/logictrace_medical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "progressive_transfer": True
    }
    
    # Run transfer learning
    trainer, history = run_medical_transfer(**config)
    
    # Load test samples
    test_data = load_medical_data(["medmcqa"])[:10]
    
    # Analyze results
    analyze_medical_transfer_results(trainer, test_data)

#!/usr/bin/env python3
"""Minimal GRPO training implementation that actually works"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from datetime import datetime
import time
import gc

def create_simple_dataset(tokenizer, num_examples=500):
    """Create a simple dataset for GRPO training"""
    # Use a simple prompt-response dataset
    prompts = []
    
    # Mathematical reasoning prompts
    math_templates = [
        "Solve step by step: {a} + {b} = ?",
        "Calculate: {a} × {b} = ?",
        "Find the value: {a} - {b} = ?",
        "What is {a} divided by {b}?",
        "Simplify: {a} + {b} - {c} = ?",
    ]
    
    # Medical reasoning prompts
    medical_templates = [
        "A patient presents with {symptom}. What is the most likely diagnosis?",
        "What is the mechanism of action of {drug}?",
        "Explain the pathophysiology of {disease}.",
        "What are the key differences between {condition1} and {condition2}?",
        "Describe the treatment approach for {condition}.",
    ]
    
    # Fill with diverse examples
    import random
    for i in range(num_examples):
        if i % 2 == 0:
            # Math example
            template = random.choice(math_templates)
            a, b, c = random.randint(1, 100), random.randint(1, 100), random.randint(1, 50)
            prompt = template.format(a=a, b=b, c=c)
        else:
            # Medical example
            template = random.choice(medical_templates)
            replacements = {
                'symptom': random.choice(['fever', 'chest pain', 'headache', 'fatigue']),
                'drug': random.choice(['aspirin', 'metformin', 'lisinopril', 'omeprazole']),
                'disease': random.choice(['diabetes', 'hypertension', 'asthma', 'COPD']),
                'condition': random.choice(['pneumonia', 'bronchitis', 'GERD', 'migraine']),
                'condition1': 'Type 1 diabetes',
                'condition2': 'Type 2 diabetes',
            }
            prompt = template.format(**replacements)
        
        prompts.append(prompt)
    
    # Tokenize prompts
    dataset_dict = []
    for prompt in prompts:
        tokens = tokenizer(prompt, truncation=True, max_length=128)
        dataset_dict.append({
            'input_ids': tokens['input_ids'],
            'query': prompt
        })
    
    return dataset_dict


def train_grpo_minimal(model_name: str, gpu_id: int = 0, num_examples: int = 500):
    """Minimal GRPO training that actually works"""
    
    # Set device
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"\n{'='*60}")
    print(f"Minimal GRPO Training")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_id}")
    print(f"Examples: {num_examples}")
    print(f"{'='*60}\n")
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use appropriate precision
    dtype = torch.float32 if "1.5B" in model_name else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Create simple dataset
    print("Creating training dataset...")
    train_dataset = create_simple_dataset(tokenizer, num_examples)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/grpo_minimal_{model_name.split('/')[-1]}_{timestamp}"
    
    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=1,
        warmup_steps=50,
        
        # GRPO specific
        kl_coef=0.05,
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95,
        
        # Memory optimization
        gradient_checkpointing=True,
        fp16=(dtype == torch.float16),
        
        # Logging
        logging_steps=10,
        save_steps=100,
        report_to=[],  # Disable wandb
    )
    
    # Simple reward function
    def reward_fn(responses, prompts):
        """Simple reward based on response quality"""
        rewards = []
        for response in responses:
            # Base reward
            reward = 0.1
            
            # Reward for length (not too short, not too long)
            length = len(response.split())
            if 10 <= length <= 100:
                reward += 0.3
            
            # Reward for reasoning indicators
            if any(word in response.lower() for word in ['because', 'therefore', 'step']):
                reward += 0.2
            
            rewards.append(reward)
        return rewards
    
    # Create trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=train_dataset,
        reward_model=reward_fn,
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    try:
        trainer.train()
        
        # Save model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        training_time = time.time() - start_time
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
        return output_dir, True
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None, False
    
    finally:
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()


def main():
    # Models to train
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0),
        ("Qwen/Qwen2.5-1.5B-Instruct", 2),
    ]
    
    print("="*80)
    print("MedLogicTrace Minimal GRPO Training")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Train each model
    results = []
    total_start = time.time()
    
    for model_name, gpu_id in models:
        output_dir, success = train_grpo_minimal(model_name, gpu_id, 500)
        results.append({
            "model": model_name,
            "success": success,
            "output_dir": output_dir
        })
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['model']}")
        if result['success']:
            print(f"   Saved to: {result['output_dir']}")
    
    # Save summary
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = f"results/grpo_minimal_{timestamp}"
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(f"{summary_dir}/training_summary.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_minutes": total_time/60,
            "results": results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fixed PPO training that works with TRL 0.17.0 based on latest API"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from datetime import datetime
import time
import json
import numpy as np
import gc
from typing import List, Dict

def create_mathematical_dataset(num_samples=500):
    """Create a dataset of mathematical reasoning problems"""
    import random
    
    queries = []
    for i in range(num_samples):
        choice = random.choice(['arithmetic', 'word_problem'])
        
        if choice == 'arithmetic':
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                query = f"Calculate step by step: {a} + {b} = ?"
            elif op == '-':
                query = f"Calculate step by step: {a} - {b} = ?"
            else:
                query = f"Calculate step by step: {a} × {b} = ?"
        else:
            # Word problem
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            templates = [
                f"John has {a} apples. Mary gives him {b} more. How many apples does John have now?",
                f"A store had {a} items. They sold {b} items. How many items are left?",
                f"There are {a} students in each class and {b} classes. How many students are there in total?"
            ]
            query = random.choice(templates)
        
        queries.append({"query": query})
    
    # Create HuggingFace Dataset
    return Dataset.from_list(queries)

def compute_reward_batch(responses: List[str], queries: List[str]) -> List[float]:
    """Compute rewards for a batch of responses"""
    rewards = []
    
    for response, query in zip(responses, queries):
        # Base reward
        reward = 0.0
        
        # Reward for step-by-step reasoning
        reasoning_keywords = ['step', 'first', 'then', 'next', 'therefore', 'so', 'thus']
        if any(keyword in response.lower() for keyword in reasoning_keywords):
            reward += 0.3
        
        # Reward for mathematical operations
        if any(op in response for op in ['=', '+', '-', '*', '×', '÷']):
            reward += 0.2
        
        # Reward for showing work
        lines = response.strip().split('\n')
        if len(lines) > 1:
            reward += 0.2
        
        # Length penalty/bonus
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            reward += 0.2
        elif word_count > 150:
            reward -= 0.2
        elif word_count < 5:
            reward -= 0.3
        
        # Bonus for numerical answer
        import re
        if re.search(r'\b\d+\b', response):
            reward += 0.1
        
        rewards.append(reward)
    
    return rewards

def train_ppo_model(model_name: str, gpu_id: int = 0, num_samples: int = 500):
    """Train a single model with PPO using correct API"""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} with PPO")
    print(f"GPU: {gpu_id}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/ppo_fixed_{model_name.replace('/', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Load model with value head
        print("Loading model with value head...")
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True
        )
        
        # Create dataset
        print("Creating dataset...")
        dataset = create_mathematical_dataset(num_samples)
        
        # Configure PPO
        print("Configuring PPO...")
        ppo_config = PPOConfig(
            learning_rate=1.41e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            
            # Key PPO parameters
            kl_coef=0.2,
            gamma=1.0,
            lam=0.95,
            cliprange=0.2,
            vf_coef=0.1,
            
            # Generation parameters
            temperature=1.0,
            
            # Training parameters
            remove_unused_columns=False,
            seed=42,
        )
        
        # Create PPO trainer with correct parameters
        print("Initializing PPO trainer...")
        ppo_trainer = PPOTrainer(
            model=model,
            config=ppo_config,
            dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Generation kwargs
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": 128,
        }
        
        print(f"Starting training with {len(dataset)} examples...")
        start_time = time.time()
        
        # Training metrics
        all_rewards = []
        all_kl_divs = []
        
        # Process batches from the dataloader
        for epoch in range(1):  # Single epoch for quick training
            print(f"\nEpoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(ppo_trainer.dataloader):
                # Tokenize queries
                query_tensors = []
                queries_text = []
                
                for query in batch["query"]:
                    queries_text.append(query)
                    encoding = tokenizer(query, return_tensors="pt", padding=False, truncation=True)
                    query_tensors.append(encoding["input_ids"][0].to(device))
                
                # Generate responses
                response_tensors = []
                responses_text = []
                
                for query_tensor in query_tensors:
                    response = ppo_trainer.generate(
                        query_tensor.unsqueeze(0),
                        return_prompt=False,
                        **generation_kwargs
                    )
                    response_tensors.append(response.squeeze())
                    response_text = tokenizer.decode(response.squeeze(), skip_special_tokens=True)
                    responses_text.append(response_text)
                
                # Compute rewards
                rewards = compute_reward_batch(responses_text, queries_text)
                reward_tensors = [torch.tensor(r).to(device) for r in rewards]
                
                # Run PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                ppo_trainer.log_stats(stats, batch, rewards)
                
                # Track metrics
                all_rewards.extend(rewards)
                if "objective/kl" in stats:
                    all_kl_divs.append(stats["objective/kl"])
                
                # Log progress
                if batch_idx % 10 == 0:
                    avg_reward = np.mean(rewards)
                    print(f"Batch {batch_idx}, Avg Reward: {avg_reward:.3f}")
                    
                    # Print example
                    if batch_idx % 50 == 0 and responses_text:
                        print(f"\nExample:")
                        print(f"Query: {queries_text[0]}")
                        print(f"Response: {responses_text[0]}")
                        print(f"Reward: {rewards[0]:.3f}\n")
        
        # Save final model
        print(f"\nSaving model to {output_dir}")
        ppo_trainer.save_pretrained(output_dir)
        
        training_time = time.time() - start_time
        
        # Save training summary
        summary = {
            "model_name": model_name,
            "gpu_id": gpu_id,
            "training_time_minutes": training_time / 60,
            "num_examples": num_samples,
            "output_dir": output_dir,
            "avg_reward": float(np.mean(all_rewards)),
            "final_kl_div": float(all_kl_divs[-1]) if all_kl_divs else 0.0,
            "reward_distribution": {
                "min": float(np.min(all_rewards)),
                "max": float(np.max(all_rewards)),
                "mean": float(np.mean(all_rewards)),
                "std": float(np.std(all_rewards))
            }
        }
        
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        print(f"✓ Average reward: {summary['avg_reward']:.3f}")
        
        # Clean up
        del model
        del ppo_trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return output_dir, True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def main():
    print("="*80)
    print("Fixed PPO Training for MedLogicTrace")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Models to train - using different GPUs
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0),
        ("Qwen/Qwen2.5-1.5B-Instruct", 2),
    ]
    
    results = []
    total_start = time.time()
    
    # Train each model
    for model_name, gpu_id in models:
        output_dir, success = train_ppo_model(model_name, gpu_id, num_samples=500)
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
    
    # Save overall summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"models/ppo_fixed_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_minutes": total_time/60,
            "results": results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return results

if __name__ == "__main__":
    results = main()

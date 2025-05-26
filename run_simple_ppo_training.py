#!/usr/bin/env python3
"""Simple PPO training that works with TRL 0.17.0"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from datetime import datetime
import time
import json
import numpy as np
import gc

# Single model at a time to avoid multiprocessing issues
def train_single_model(model_name: str, gpu_id: int, num_samples: int = 500):
    """Train a single model with PPO"""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} on GPU {gpu_id}")
    print(f"{'='*60}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/ppo_{model_name.replace('/', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Determine precision
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        
        # Load model with value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True
        )
        
        # Create simple dataset
        queries = []
        for i in range(num_samples):
            a, b = np.random.randint(1, 100), np.random.randint(1, 100)
            query = f"Calculate step by step: {a} + {b} = ?"
            queries.append(query)
        
        # Create dataset
        dataset = Dataset.from_dict({"query": queries})
        
        # Configure PPO
        ppo_config = PPOConfig(
            batch_size=16,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            kl_coef=0.05,
            cliprange=0.2,
            vf_coef=0.1,
            gamma=0.99,
            lam=0.95,
            temperature=0.8,
            seed=42,
        )
        
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
        )
        
        # Simple reward function
        def compute_rewards(responses):
            rewards = []
            for response in responses:
                # Basic reward based on response quality
                reward = 0.1
                if "=" in response:
                    reward += 0.3
                if len(response.split()) > 5:
                    reward += 0.2
                rewards.append(reward)
            return rewards
        
        print(f"Starting training with {num_samples} examples...")
        start_time = time.time()
        
        # Training loop
        all_stats = []
        batch_size = 16
        
        for epoch in range(1):  # Single epoch for quick training
            print(f"\nEpoch {epoch+1}")
            
            for batch_idx in range(0, len(queries), batch_size):
                batch_queries = queries[batch_idx:batch_idx+batch_size]
                
                # Tokenize queries
                query_tensors = []
                for q in batch_queries:
                    tokens = tokenizer.encode(q, return_tensors="pt")
                    query_tensors.append(tokens[0])
                
                # Generate responses
                response_tensors = []
                for query_tensor in query_tensors:
                    with torch.no_grad():
                        response = ppo_trainer.generate(
                            query_tensor.unsqueeze(0).to(device),
                            max_new_tokens=64,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    response_tensors.append(response.squeeze()[len(query_tensor):])
                
                # Decode responses
                responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                
                # Calculate rewards
                rewards = compute_rewards(responses)
                reward_tensors = [torch.tensor(r).to(device) for r in rewards]
                
                # PPO step
                try:
                    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                    all_stats.append(stats)
                    
                    if batch_idx % 50 == 0:
                        avg_reward = np.mean(rewards)
                        print(f"Batch {batch_idx//batch_size}, Avg Reward: {avg_reward:.3f}")
                except Exception as e:
                    print(f"Error in PPO step: {e}")
                    continue
        
        # Save model
        print(f"\nSaving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        training_time = time.time() - start_time
        
        # Save summary
        summary = {
            "model_name": model_name,
            "gpu_id": gpu_id,
            "training_time_minutes": training_time / 60,
            "num_examples": num_samples,
            "output_dir": output_dir,
            "final_stats": all_stats[-1] if all_stats else {}
        }
        
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        
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
    print("PPO Training for MedLogicTrace")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Models to train
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0),
        ("Qwen/Qwen2.5-1.5B-Instruct", 2),
        ("Qwen/Qwen2.5-0.5B", 4),
        ("Qwen/Qwen2.5-1.5B", 6),
    ]
    
    results = []
    
    # Train each model sequentially
    for model_name, gpu_id in models:
        output_dir, success = train_single_model(model_name, gpu_id, num_samples=500)
        results.append({
            "model": model_name,
            "success": success,
            "output_dir": output_dir
        })
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['model']}")
        if result['success']:
            print(f"   Saved to: {result['output_dir']}")
    
    # Save overall summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"models/ppo_training_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    return results


if __name__ == "__main__":
    results = main()

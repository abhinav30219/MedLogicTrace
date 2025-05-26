#!/usr/bin/env python3
"""Parallel GRPO/PPO training on 8 GPUs with checkpointing"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import json
import numpy as np
from typing import List, Dict, Tuple
import gc

# Models to train with GRPO
MODELS_TO_TRAIN = [
    ("Qwen/Qwen2.5-0.5B-Instruct", [0, 1]),      # GPUs 0-1
    ("Qwen/Qwen2.5-1.5B-Instruct", [2, 3]),      # GPUs 2-3
    ("Qwen/Qwen2.5-0.5B", [4, 5]),               # GPUs 4-5 (base model)
    ("Qwen/Qwen2.5-1.5B", [6, 7]),               # GPUs 6-7 (base model)
]

TRAIN_SAMPLES = 2000  # Number of training samples
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 1


def load_training_data(num_samples=2000):
    """Load mathematical reasoning data for training"""
    try:
        # Try to load GSM8K dataset
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Format for PPO training
        formatted_data = []
        for item in dataset:
            question = item['question']
            answer = item['answer']
            # Create query for the model
            query = f"Question: {question}\nLet's solve this step by step:\n"
            formatted_data.append({
                'query': query,
                'answer': answer
            })
        
        return formatted_data
    except:
        # Fallback to synthetic data
        print("Using synthetic mathematical data...")
        data = []
        import random
        
        for i in range(num_samples):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            operation = random.choice(['+', '-', '*'])
            
            if operation == '+':
                result = a + b
                query = f"Calculate step by step: {a} + {b} = ?"
            elif operation == '-':
                result = a - b
                query = f"Calculate step by step: {a} - {b} = ?"
            else:
                result = a * b
                query = f"Calculate step by step: {a} × {b} = ?"
            
            data.append({
                'query': query,
                'answer': str(result)
            })
        
        return data


def create_reward_function():
    """Create reward function for mathematical reasoning"""
    def reward_fn(query_response_pairs):
        """Calculate rewards based on reasoning quality"""
        rewards = []
        
        for query, response in query_response_pairs:
            # Base reward
            reward = 0.0
            
            # Check for step-by-step reasoning
            if any(indicator in response.lower() for indicator in 
                   ['step', 'first', 'then', 'next', 'finally', 'therefore']):
                reward += 0.3
            
            # Check for mathematical operations
            if any(op in response for op in ['=', '+', '-', '*', '/', '×']):
                reward += 0.2
            
            # Length penalty/bonus
            word_count = len(response.split())
            if 20 <= word_count <= 100:
                reward += 0.2
            elif word_count > 150:
                reward -= 0.1
            
            # Penalize very short responses
            if word_count < 10:
                reward -= 0.3
            
            rewards.append(reward)
        
        return rewards
    
    return reward_fn


def train_model_on_gpus(model_name: str, gpu_ids: List[int], train_data: List[Dict], 
                       checkpoint_dir: str) -> Dict:
    """Train a single model using multiple GPUs"""
    import torch
    import os
    
    # Set primary GPU
    torch.cuda.set_device(gpu_ids[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} on GPUs {gpu_ids}")
    print(f"{'='*60}\n")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Important for generation
        
        # Determine precision
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        
        # Load model with value head for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if len(gpu_ids) > 1 else {"": f"cuda:{gpu_ids[0]}"},
            trust_remote_code=True
        )
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{checkpoint_dir}/{model_name.replace('/', '_')}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure PPO (similar to GRPO)
        ppo_config = PPOConfig(
            batch_size=BATCH_SIZE * len(gpu_ids),
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=1e-5,
            
            # PPO/GRPO specific
            kl_coef=0.05,      # KL penalty coefficient
            cliprange=0.2,     # PPO clip range
            vf_coef=0.1,       # Value function coefficient
            gamma=0.99,        # Discount factor
            lam=0.95,          # GAE lambda
            
            # Generation
            temperature=0.8,
            
            # Training
            seed=42,
            remove_unused_columns=False,
        )
        
        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
        )
        
        # Prepare queries
        queries = [item['query'] for item in train_data]
        query_tensors = [tokenizer.encode(q, return_tensors="pt")[0] for q in queries]
        
        # Training loop
        reward_fn = create_reward_function()
        
        print(f"Starting training with {len(queries)} examples...")
        start_time = time.time()
        
        # Process in batches
        batch_size = BATCH_SIZE * len(gpu_ids)
        num_batches = len(queries) // batch_size
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            
            for batch_idx in range(num_batches):
                # Get batch
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(queries))
                batch_queries = query_tensors[batch_start:batch_end]
                
                # Generate responses
                try:
                    response_tensors = []
                    for query in batch_queries:
                        gen_len = len(query)
                        response = ppo_trainer.generate(
                            query.unsqueeze(0).to(model.device),
                            max_new_tokens=128,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        response_tensors.append(response.squeeze()[gen_len:])
                    
                    # Decode responses
                    responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                    
                    # Calculate rewards
                    query_response_pairs = list(zip(queries[batch_start:batch_end], responses))
                    rewards = reward_fn(query_response_pairs)
                    reward_tensors = [torch.tensor(r) for r in rewards]
                    
                    # PPO step
                    stats = ppo_trainer.step(batch_queries, response_tensors, reward_tensors)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}/{num_batches}, "
                              f"Avg Reward: {np.mean(rewards):.3f}, "
                              f"KL: {stats['objective/kl']:.3f}")
                    
                    # Save checkpoint every 100 batches
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        checkpoint_path = f"{output_dir}/checkpoint-{epoch}-{batch_idx}"
                        ppo_trainer.save_pretrained(checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Save final model
        final_path = f"{output_dir}/final_model"
        ppo_trainer.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        training_time = time.time() - start_time
        
        # Save training summary
        summary = {
            "model_name": model_name,
            "gpu_ids": gpu_ids,
            "training_time_minutes": training_time / 60,
            "num_examples": len(queries),
            "final_model_path": final_path,
            "checkpoints": [f for f in os.listdir(output_dir) if f.startswith("checkpoint")]
        }
        
        with open(f"{output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        print(f"✓ Model saved to {final_path}")
        
        # Clean up
        del model
        del ppo_trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return summary
        
    except Exception as e:
        print(f"✗ Training failed for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Set multiprocessing start method for CUDA
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    print("="*80)
    print("Parallel GRPO/PPO Training on 8 GPUs")
    print(f"Started: {datetime.now()}")
    print(f"Models: {len(MODELS_TO_TRAIN)}")
    print(f"Training samples: {TRAIN_SAMPLES}")
    print("="*80)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"models/grpo_parallel_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load training data
    print("\nLoading training data...")
    train_data = load_training_data(TRAIN_SAMPLES)
    print(f"Loaded {len(train_data)} training examples")
    
    # Train models in parallel
    print("\nStarting parallel training...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=len(MODELS_TO_TRAIN)) as executor:
        # Submit training jobs
        futures = {}
        for model_name, gpu_ids in MODELS_TO_TRAIN:
            future = executor.submit(
                train_model_on_gpus,
                model_name, gpu_ids, train_data, checkpoint_dir
            )
            futures[future] = model_name
        
        # Collect results
        results = []
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"\n✓ Completed training for {model_name}")
            except Exception as e:
                print(f"\n✗ Training failed for {model_name}: {e}")
    
    total_time = time.time() - start_time
    
    # Save overall summary
    overall_summary = {
        "timestamp": timestamp,
        "total_time_minutes": total_time / 60,
        "num_models": len(MODELS_TO_TRAIN),
        "successful_models": len(results),
        "checkpoint_dir": checkpoint_dir,
        "model_results": results
    }
    
    with open(f"{checkpoint_dir}/overall_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("PARALLEL TRAINING COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {len(results)}/{len(MODELS_TO_TRAIN)} models")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Print model paths
    print("\nTrained model paths:")
    for result in results:
        if result:
            print(f"  - {result['model_name']}: {result['final_model_path']}")
    
    return checkpoint_dir, results


if __name__ == "__main__":
    checkpoint_dir, results = main()

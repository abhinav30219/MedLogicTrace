#!/usr/bin/env python3
"""Custom GRPO implementation without TRL for reasoning enhancement"""

import os
import sys
sys.path.append('src')

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from datetime import datetime
import time
import json
import numpy as np
import gc
from typing import List, Dict, Tuple
import random

def create_reasoning_dataset(num_samples=1000):
    """Create dataset of problems requiring step-by-step reasoning"""
    problems = []
    
    # Mathematical reasoning problems
    for i in range(num_samples // 2):
        problem_type = random.choice(['arithmetic', 'word_problem', 'algebra'])
        
        if problem_type == 'arithmetic':
            a = random.randint(10, 100)
            b = random.randint(10, 100)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                problem = f"Calculate step by step: {a} + {b}"
                answer = a + b
            elif op == '-':
                problem = f"Calculate step by step: {a} - {b}"
                answer = a - b
            else:
                problem = f"Calculate step by step: {a} × {b}"
                answer = a * b
                
        elif problem_type == 'word_problem':
            a = random.randint(5, 50)
            b = random.randint(5, 50)
            templates = [
                (f"A store has {a} apples. They receive {b} more. How many apples in total?", a + b),
                (f"There are {a} students. {b} leave. How many remain?", a - b),
                (f"Each box has {a} items. There are {b} boxes. How many items total?", a * b),
            ]
            problem, answer = random.choice(templates)
            
        else:  # algebra
            a = random.randint(2, 20)
            b = random.randint(10, 50)
            problem = f"Solve for x: {a}x = {b}"
            answer = b / a
        
        problems.append({
            'prompt': f"Problem: {problem}\nShow your reasoning step by step:",
            'answer': str(answer),
            'type': problem_type
        })
    
    # Medical reasoning problems
    medical_scenarios = [
        ("Patient has fever and cough. What are possible diagnoses?", "respiratory infection"),
        ("Blood pressure 160/100. What type of hypertension?", "stage 2 hypertension"),
        ("Patient on warfarin, INR 5.5. What action needed?", "hold warfarin, vitamin K"),
        ("Chest pain with ST elevation. Diagnosis?", "STEMI, acute MI"),
        ("Sudden headache, stiff neck, photophobia. Consider?", "meningitis"),
    ]
    
    for _ in range(num_samples // 2):
        scenario, key_point = random.choice(medical_scenarios)
        problems.append({
            'prompt': f"Medical case: {scenario}\nProvide step-by-step clinical reasoning:",
            'answer': key_point,
            'type': 'medical'
        })
    
    return problems

def compute_reasoning_reward(response: str, problem_type: str, expected_answer: str = None) -> float:
    """Compute reward based on reasoning quality"""
    reward = 0.0
    
    # Base reward for attempting reasoning
    reward += 0.1
    
    # Reward for step-by-step structure
    lines = response.strip().split('\n')
    if len(lines) > 1:
        reward += 0.2
    
    # Reward for reasoning indicators
    reasoning_words = ['step', 'first', 'then', 'next', 'therefore', 'because', 'since', 'thus']
    for word in reasoning_words:
        if word.lower() in response.lower():
            reward += 0.05
    
    # Reward for mathematical operations (if applicable)
    if problem_type in ['arithmetic', 'word_problem', 'algebra']:
        math_symbols = ['=', '+', '-', '*', '×', '÷', '/', 'x']
        for symbol in math_symbols:
            if symbol in response:
                reward += 0.05
    
    # Reward for medical terminology (if applicable)
    if problem_type == 'medical':
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'examination', 'history', 'differential']
        for term in medical_terms:
            if term.lower() in response.lower():
                reward += 0.05
    
    # Length penalty/bonus
    word_count = len(response.split())
    if 20 <= word_count <= 150:
        reward += 0.2
    elif word_count > 200:
        reward -= 0.1
    elif word_count < 10:
        reward -= 0.3
    
    # Check if answer is mentioned (if provided)
    if expected_answer and expected_answer.lower() in response.lower():
        reward += 0.3
    
    # Ensure reward is in reasonable range
    return max(0.0, min(1.0, reward))

def generate_response_group(model, tokenizer, prompt: str, K: int = 4, device: str = "cuda") -> List[str]:
    """Generate K diverse responses for a prompt"""
    responses = []
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    for i in range(K):
        # Vary generation parameters for diversity
        temperature = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0
        top_p = 0.9 - (i * 0.05)      # 0.9, 0.85, 0.8, 0.75
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses

def compute_log_probs(model, tokenizer, prompt: str, response: str, device: str = "cuda") -> torch.Tensor:
    """Compute log probabilities of response given prompt"""
    # Combine prompt and response
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Get prompt length
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_length = prompt_inputs['input_ids'].shape[1]
    
    # Don't use torch.no_grad() here - we need gradients for training!
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs.logits
    
    # Get log probs for response tokens only
    response_logits = logits[:, prompt_length-1:-1, :]
    response_labels = inputs['input_ids'][:, prompt_length:]
    
    # Compute log probabilities
    log_probs = F.log_softmax(response_logits, dim=-1)
    response_log_probs = log_probs.gather(2, response_labels.unsqueeze(-1)).squeeze(-1)
    
    return response_log_probs.mean()

def grpo_training_step(model, tokenizer, optimizer, problems: List[Dict], K: int = 4, device: str = "cuda"):
    """Single GRPO training step"""
    total_loss = 0.0
    total_reward = 0.0
    num_updates = 0
    
    model.train()
    
    for problem in problems:
        prompt = problem['prompt']
        problem_type = problem['type']
        expected_answer = problem.get('answer', None)
        
        # Generate K responses
        responses = generate_response_group(model, tokenizer, prompt, K, device)
        
        # Compute rewards
        rewards = []
        for response in responses:
            reward = compute_reasoning_reward(response, problem_type, expected_answer)
            rewards.append(reward)
        
        rewards = np.array(rewards)
        total_reward += rewards.mean()
        
        # Normalize rewards within group (GRPO key insight)
        if rewards.std() > 0:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()
        
        # Compute loss for each response
        for response, norm_reward in zip(responses, normalized_rewards):
            if abs(norm_reward) > 0.01:  # Only update on meaningful differences
                log_prob = compute_log_probs(model, tokenizer, prompt, response, device)
                loss = -log_prob * norm_reward
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
    
    avg_loss = total_loss / max(num_updates, 1)
    avg_reward = total_reward / len(problems)
    
    return avg_loss, avg_reward

def evaluate_reasoning(model, tokenizer, test_problems: List[Dict], device: str = "cuda") -> Dict:
    """Evaluate model's reasoning capabilities"""
    model.eval()
    
    total_reward = 0.0
    results_by_type = {}
    example_outputs = []
    
    for i, problem in enumerate(test_problems):
        prompt = problem['prompt']
        problem_type = problem['type']
        expected_answer = problem.get('answer', None)
        
        # Generate single response for evaluation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        reward = compute_reasoning_reward(response, problem_type, expected_answer)
        
        total_reward += reward
        
        if problem_type not in results_by_type:
            results_by_type[problem_type] = []
        results_by_type[problem_type].append(reward)
        
        # Save examples
        if i < 5:  # Save first 5 examples
            example_outputs.append({
                'prompt': prompt,
                'response': response,
                'reward': reward,
                'type': problem_type
            })
    
    # Compute statistics
    results = {
        'avg_reward': total_reward / len(test_problems),
        'results_by_type': {k: np.mean(v) for k, v in results_by_type.items()},
        'example_outputs': example_outputs
    }
    
    return results

def train_grpo_model(model_name: str, gpu_id: int = 0, num_epochs: int = 3, K: int = 4, custom_train_data=None):
    """Main GRPO training function"""
    # When CUDA_VISIBLE_DEVICES is set, always use device 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Custom GRPO Training")
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_id}")
    print(f"Epochs: {num_epochs}")
    print(f"Group size (K): {K}")
    print(f"{'='*60}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/grpo_custom_{model_name.replace('/', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.float32 if "1.5B" in model_name else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # Create datasets
    print("Creating datasets...")
    if custom_train_data is not None:
        train_problems = custom_train_data
        print(f"Using custom training data with {len(train_problems)} samples")
    else:
        train_problems = create_reasoning_dataset(100)  # Reduced from 500
    test_problems = create_reasoning_dataset(20)    # Reduced from 100
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Evaluate before training
    print("Initial evaluation...")
    initial_results = evaluate_reasoning(model, tokenizer, test_problems[:50], device)
    print(f"Initial average reward: {initial_results['avg_reward']:.3f}")
    
    # Training loop
    print("\nStarting GRPO training...")
    training_history = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        epoch_reward = 0.0
        
        # Process in batches
        batch_size = 8
        num_batches = len(train_problems) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(train_problems))
            batch_problems = train_problems[batch_start:batch_end]
            
            # GRPO training step
            loss, reward = grpo_training_step(model, tokenizer, optimizer, batch_problems, K, device)
            
            epoch_loss += loss
            epoch_reward += reward
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, Avg Reward: {reward:.3f}")
        
        # Epoch statistics
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_reward = epoch_reward / num_batches
        
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}, Avg Reward: {avg_epoch_reward:.3f}")
        
        # Evaluate after each epoch
        eval_results = evaluate_reasoning(model, tokenizer, test_problems[:50], device)
        print(f"Evaluation - Avg Reward: {eval_results['avg_reward']:.3f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_epoch_loss,
            'train_reward': avg_epoch_reward,
            'eval_reward': eval_results['avg_reward']
        })
        
        # Save checkpoint
        checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    # Final evaluation
    print("\nFinal evaluation on full test set...")
    final_results = evaluate_reasoning(model, tokenizer, test_problems, device)
    
    training_time = time.time() - start_time
    
    # Save final model and results
    print(f"\nSaving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training summary
    summary = {
        'model_name': model_name,
        'training_time_minutes': training_time / 60,
        'num_epochs': num_epochs,
        'group_size_K': K,
        'initial_reward': initial_results['avg_reward'],
        'final_reward': final_results['avg_reward'],
        'improvement': final_results['avg_reward'] - initial_results['avg_reward'],
        'final_results_by_type': final_results['results_by_type'],
        'training_history': training_history,
        'example_outputs': final_results['example_outputs']
    }
    
    with open(f"{output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
    print(f"✓ Initial reward: {initial_results['avg_reward']:.3f}")
    print(f"✓ Final reward: {final_results['avg_reward']:.3f}")
    print(f"✓ Improvement: {summary['improvement']:.3f}")
    
    # Print example outputs
    print("\nExample outputs:")
    for i, example in enumerate(final_results['example_outputs'][:2]):
        print(f"\n--- Example {i+1} ({example['type']}) ---")
        print(f"Prompt: {example['prompt']}")
        print(f"Response: {example['response']}")
        print(f"Reward: {example['reward']:.3f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_dir, summary

def main():
    print("="*80)
    print("Custom GRPO Training for Reasoning Enhancement")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Models to train
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", 0),
        ("Qwen/Qwen2.5-1.5B-Instruct", 2),
    ]
    
    results = []
    total_start = time.time()
    
    # Train each model
    for model_name, gpu_id in models:
        try:
            output_dir, summary = train_grpo_model(model_name, gpu_id, num_epochs=2, K=4)
            results.append({
                'model': model_name,
                'success': True,
                'output_dir': output_dir,
                'summary': summary
            })
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results.append({
                'model': model_name,
                'success': False,
                'error': str(e)
            })
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "="*80)
    print("GRPO TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"\n{status} {result['model']}")
        if result['success']:
            summary = result['summary']
            print(f"   Output: {result['output_dir']}")
            print(f"   Improvement: {summary['improvement']:.3f}")
            print(f"   Final reward: {summary['final_reward']:.3f}")
    
    # Save overall summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"models/grpo_custom_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({
            'timestamp': timestamp,
            'total_time_minutes': total_time/60,
            'results': results
        }, f, indent=2)
    
    print(f"\nOverall summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate GRPO-trained models on medical reasoning tasks"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from datetime import datetime
import numpy as np
from typing import Dict, List
from data_utils import load_medical_dataset, format_medical_prompt, evaluate_medical_response

def find_latest_grpo_models(models_dir: str) -> List[str]:
    """Find the latest GRPO-trained model directories"""
    model_dirs = []
    
    if not os.path.exists(models_dir):
        return []
    
    for dir_name in os.listdir(models_dir):
        if dir_name.startswith("grpo_custom_") and os.path.isdir(os.path.join(models_dir, dir_name)):
            # Check if it has model files
            dir_path = os.path.join(models_dir, dir_name)
            if os.path.exists(os.path.join(dir_path, "config.json")) or \
               os.path.exists(os.path.join(dir_path, "checkpoint_epoch_1", "config.json")):
                model_dirs.append(dir_path)
    
    # Sort by timestamp in filename
    model_dirs.sort(reverse=True)
    return model_dirs[:8]  # Get the 8 most recent models

def evaluate_grpo_model(model_path: str, device: str = "cuda:0") -> Dict:
    """Evaluate a single GRPO model"""
    print(f"\nEvaluating model: {os.path.basename(model_path)}")
    
    # Check for checkpoint directory
    if os.path.exists(os.path.join(model_path, "checkpoint_epoch_1")):
        model_path = os.path.join(model_path, "checkpoint_epoch_1")
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": device}
        )
        model.eval()
        
        # Extract model name from path
        base_path = os.path.basename(model_path)
        if "checkpoint_epoch" in base_path:
            base_path = os.path.basename(os.path.dirname(model_path))
        
        model_name = base_path.replace("grpo_custom_", "").replace("_", "/")
        for timestamp in ["20250526_062523", "20250526_062525", "20250526_062527", "20250526_062529", 
                         "20250526_062531", "20250526_062533", "20250526_062535", "20250526_062536"]:
            model_name = model_name.replace(f"/{timestamp}", "")
        
        # Evaluate on medical datasets
        results = {'model_name': model_name, 'model_path': model_path}
        
        for dataset_name in ['medmcqa', 'pubmedqa']:
            print(f"\nEvaluating on {dataset_name}...")
            examples = load_medical_dataset(dataset_name, num_samples=100)
            
            correct = 0
            total_tokens = 0
            num_responses = 0
            
            for i, example in enumerate(examples):
                if i % 20 == 0:
                    print(f"  Progress: {i}/{len(examples)}")
                
                # Format prompt
                prompt = format_medical_prompt(example)
                
                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # Count tokens
                response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                total_tokens += response_tokens
                num_responses += 1
                
                # Evaluate correctness
                if evaluate_medical_response(response, example):
                    correct += 1
            
            # Store results
            accuracy = correct / len(examples) if examples else 0
            avg_tokens = total_tokens / num_responses if num_responses > 0 else 0
            
            results[f'{dataset_name}_accuracy'] = accuracy
            results[f'{dataset_name}_avg_tokens'] = avg_tokens
            
            print(f"  {dataset_name} - Accuracy: {accuracy:.1%}, Avg tokens: {avg_tokens:.1f}")
        
        # Calculate overall metrics
        results['overall_accuracy'] = (results['medmcqa_accuracy'] + results['pubmedqa_accuracy']) / 2
        results['overall_avg_tokens'] = (results['medmcqa_avg_tokens'] + results['pubmedqa_avg_tokens']) / 2
        results['efficiency_score'] = results['overall_accuracy'] / (results['overall_avg_tokens'] / 100)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("GRPO Model Evaluation on Medical Reasoning Tasks")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Find GRPO models
    models_dir = "models"
    grpo_models = find_latest_grpo_models(models_dir)
    
    print(f"\nFound {len(grpo_models)} GRPO models to evaluate")
    
    # Evaluate each model
    results = []
    for model_path in grpo_models:
        result = evaluate_grpo_model(model_path, device="cuda:0")
        if result:
            results.append(result)
            print(f"\nResults for {result['model_name']}:")
            print(f"  Overall Accuracy: {result['overall_accuracy']:.1%}")
            print(f"  Average Tokens: {result['overall_avg_tokens']:.1f}")
            print(f"  Efficiency Score: {result['efficiency_score']:.2f}")
    
    # Compare with baseline results
    baseline_results = {
        "Qwen/Qwen2.5-0.5B": {"accuracy": 0.635, "tokens": 29.9, "efficiency": 2.12},
        "Qwen/Qwen2.5-1.5B": {"accuracy": 0.598, "tokens": 21.5, "efficiency": 2.78},
        "Qwen/Qwen2.5-0.5B-Instruct": {"accuracy": 0.843, "tokens": 64.0, "efficiency": 1.32},
        "Qwen/Qwen2.5-1.5B-Instruct": {"accuracy": 0.819, "tokens": 64.0, "efficiency": 1.28}
    }
    
    # Create comparison summary
    print("\n" + "="*80)
    print("COMPARISON: GRPO vs Baseline Models")
    print("="*80)
    
    for result in results:
        model_key = result['model_name'].split('_')[0]  # Get base model name
        if model_key in baseline_results:
            baseline = baseline_results[model_key]
            accuracy_change = (result['overall_accuracy'] - baseline['accuracy']) * 100
            token_change = result['overall_avg_tokens'] - baseline['tokens']
            efficiency_change = result['efficiency_score'] - baseline['efficiency']
            
            print(f"\n{result['model_name']}:")
            print(f"  Accuracy: {result['overall_accuracy']:.1%} ({accuracy_change:+.1f}% vs baseline)")
            print(f"  Tokens: {result['overall_avg_tokens']:.1f} ({token_change:+.1f} vs baseline)")
            print(f"  Efficiency: {result['efficiency_score']:.2f} ({efficiency_change:+.2f} vs baseline)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/grpo_evaluation_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'grpo_results': results,
            'baseline_results': baseline_results,
            'summary': {
                'models_evaluated': len(results),
                'average_accuracy_improvement': np.mean([r['overall_accuracy'] - baseline_results.get(r['model_name'].split('_')[0], {}).get('accuracy', 0) 
                                                        for r in results if r['model_name'].split('_')[0] in baseline_results]),
                'average_token_change': np.mean([r['overall_avg_tokens'] - baseline_results.get(r['model_name'].split('_')[0], {}).get('tokens', 0) 
                                               for r in results if r['model_name'].split('_')[0] in baseline_results])
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()

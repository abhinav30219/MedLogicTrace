#!/usr/bin/env python3
"""Evaluate GRPO models and prepare for local transfer"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import json
import subprocess
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np


def evaluate_grpo_model(model_path: str, model_name: str, gpu_id: int) -> List[Dict]:
    """Evaluate a GRPO-trained model on medical datasets"""
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    from src.data_utils import load_medical_dataset, format_medical_prompt
    
    print(f"[GPU {gpu_id}] Evaluating {model_name}")
    
    results = []
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine precision
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True
        )
        model.eval()
        
        # Evaluate on each dataset
        for dataset_name in ["medmcqa", "pubmed_qa"]:
            print(f"[GPU {gpu_id}] Evaluating on {dataset_name}")
            
            examples = load_medical_dataset(dataset_name, 500)
            
            correct = 0
            total = 0
            token_counts = []
            generation_failures = 0
            
            for i, example in enumerate(examples):
                if i % 100 == 0:
                    print(f"[GPU {gpu_id}] Progress: {i}/500")
                
                prompt = format_medical_prompt(example)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=64,
                            temperature=0.8,
                            do_sample=True,
                            top_p=0.95,
                            top_k=50,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                        
                        if "options" in example and response:
                            answer_letter = chr(65 + example["answer"])
                            if answer_letter.lower() in response.lower()[:50]:
                                correct += 1
                            token_counts.append(len(tokenizer.encode(response)))
                        
                        total += 1
                        
                    except Exception as e:
                        generation_failures += 1
                        total += 1
            
            accuracy = correct / total if total > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            
            result = {
                "model": model_path,
                "model_name": model_name,
                "model_type": "grpo",
                "dataset": dataset_name,
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "total_examples": total,
                "generation_failures": generation_failures,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"[GPU {gpu_id}] {model_name} on {dataset_name}: {accuracy:.2%} accuracy, {avg_tokens:.1f} tokens")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error evaluating {model_name}: {e}")
        return []


def create_comparison_plots(baseline_df, grpo_df, output_dir):
    """Create comparison plots between baseline and GRPO models"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy comparison
    models = []
    baseline_acc = []
    grpo_acc = []
    
    for model_type in ['0.5B-Instruct', '1.5B-Instruct', '0.5B', '1.5B']:
        # Find matching models
        base_mask = baseline_df['model'].str.contains(model_type.replace('-Instruct', ''))
        if '-Instruct' in model_type:
            base_mask &= baseline_df['model_type'].str.contains('instruct')
        else:
            base_mask &= baseline_df['model_type'].str.contains('base')
        
        grpo_mask = grpo_df['model_name'].str.contains(model_type.replace('-Instruct', ''))
        if '-Instruct' in model_type:
            grpo_mask &= grpo_df['model_name'].str.contains('Instruct')
        else:
            grpo_mask &= ~grpo_df['model_name'].str.contains('Instruct')
        
        if base_mask.any() and grpo_mask.any():
            models.append(model_type)
            baseline_acc.append(baseline_df[base_mask]['accuracy'].mean())
            grpo_acc.append(grpo_df[grpo_mask]['accuracy'].mean())
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline', color='lightblue')
    bars2 = ax1.bar(x + width/2, grpo_acc, width, label='GRPO', color='darkblue')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy: Baseline vs GRPO')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
    
    # 2. Token efficiency comparison
    baseline_tokens = []
    grpo_tokens = []
    
    for model in models:
        base_mask = baseline_df['model'].str.contains(model.split('-')[0])
        grpo_mask = grpo_df['model_name'].str.contains(model.split('-')[0])
        
        if base_mask.any():
            baseline_tokens.append(baseline_df[base_mask]['avg_tokens'].mean())
        if grpo_mask.any():
            grpo_tokens.append(grpo_df[grpo_mask]['avg_tokens'].mean())
    
    bars1 = ax2.bar(x - width/2, baseline_tokens, width, label='Baseline', color='lightgreen')
    bars2 = ax2.bar(x + width/2, grpo_tokens, width, label='GRPO', color='darkgreen')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Tokens')
    ax2.set_title('Token Usage: Baseline vs GRPO')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    
    # 3. Efficiency score
    efficiency_baseline = [a / (t / 100) for a, t in zip(baseline_acc, baseline_tokens)]
    efficiency_grpo = [a / (t / 100) for a, t in zip(grpo_acc, grpo_tokens)]
    
    bars1 = ax3.bar(x - width/2, efficiency_baseline, width, label='Baseline', color='lightyellow')
    bars2 = ax3.bar(x + width/2, efficiency_grpo, width, label='GRPO', color='gold')
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Efficiency Score (Accuracy/Tokens×100): Baseline vs GRPO')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    
    # 4. Improvement percentage
    acc_improvement = [(g - b) / b * 100 for g, b in zip(grpo_acc, baseline_acc)]
    token_reduction = [(b - g) / b * 100 for g, b in zip(grpo_tokens, baseline_tokens)]
    
    x2 = np.arange(len(models))
    bars1 = ax4.bar(x2 - width/2, acc_improvement, width, label='Accuracy Improvement %', color='green')
    bars2 = ax4.bar(x2 + width/2, token_reduction, width, label='Token Reduction %', color='red')
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Improvement %')
    ax4.set_title('GRPO Improvements over Baseline')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle('GRPO Training Results Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/grpo_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {output_dir}/grpo_comparison_plot.png")


def main():
    """Main function to evaluate GRPO models and prepare for transfer"""
    print("="*80)
    print("GRPO Model Evaluation and Transfer Preparation")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Find the latest GRPO checkpoint directory
    model_dirs = [d for d in os.listdir('models') if d.startswith('grpo_parallel_')]
    if not model_dirs:
        print("No GRPO model directories found!")
        return
    
    latest_dir = sorted(model_dirs)[-1]
    checkpoint_dir = f"models/{latest_dir}"
    print(f"\nUsing checkpoint directory: {checkpoint_dir}")
    
    # Load training summary
    with open(f"{checkpoint_dir}/overall_summary.json", "r") as f:
        training_summary = json.load(f)
    
    # Prepare models for evaluation
    models_to_evaluate = []
    for result in training_summary['model_results']:
        if result and 'final_model_path' in result:
            models_to_evaluate.append({
                'path': result['final_model_path'],
                'name': result['model_name']
            })
    
    print(f"\nFound {len(models_to_evaluate)} GRPO models to evaluate")
    
    # Evaluate models in parallel
    print("\nStarting parallel evaluation...")
    start_time = datetime.now()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=min(len(models_to_evaluate), 4)) as executor:
        futures = {}
        for i, model_info in enumerate(models_to_evaluate):
            gpu_id = i % torch.cuda.device_count()
            future = executor.submit(
                evaluate_grpo_model,
                model_info['path'],
                model_info['name'],
                gpu_id
            )
            futures[future] = model_info['name']
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    print(f"✓ Evaluation completed for {model_name}")
            except Exception as e:
                print(f"✗ Evaluation failed for {model_name}: {e}")
    
    eval_time = (datetime.now() - start_time).total_seconds() / 60
    
    # Save evaluation results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = f"{checkpoint_dir}/grpo_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nEvaluation results saved to {results_path}")
        
        # Load baseline results for comparison
        baseline_path = "MedLogicTrace/results/baseline_results.csv"
        if os.path.exists(baseline_path):
            baseline_df = pd.read_csv(baseline_path)
            
            # Create comparison plots
            create_comparison_plots(baseline_df, results_df, checkpoint_dir)
            
            # Print summary
            print("\nEVALUATION SUMMARY:")
            print("-"*50)
            for model_type in results_df['model_name'].unique():
                model_results = results_df[results_df['model_name'] == model_type]
                avg_acc = model_results['accuracy'].mean()
                avg_tokens = model_results['avg_tokens'].mean()
                print(f"{model_type}:")
                print(f"  Accuracy: {avg_acc:.2%}")
                print(f"  Avg Tokens: {avg_tokens:.1f}")
    
    # Create a compressed archive for transfer
    print("\nCreating compressed archive for transfer...")
    archive_name = f"grpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    
    # Create archive with all GRPO models and results
    cmd = f"cd models && tar -czf ../{archive_name} {latest_dir}"
    subprocess.run(cmd, shell=True)
    
    print(f"\n✓ Archive created: {archive_name}")
    print(f"✓ Total evaluation time: {eval_time:.1f} minutes")
    
    # Print transfer instructions
    print("\nTo transfer the GRPO models to your local machine, run:")
    print(f"scp -P 22046 -i ~/.ssh/id_ed25519 root@63.141.33.3:/workspace/MedLogicTrace/{archive_name} .")
    
    return checkpoint_dir, archive_name


if __name__ == "__main__":
    checkpoint_dir, archive_name = main()

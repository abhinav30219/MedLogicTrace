#!/usr/bin/env python3
"""Parallel GPU experiments: baseline evaluation, GRPO training, and post-training evaluation"""

import os
import sys
sys.path.append('src')

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import json
import subprocess
import time

# Models to experiment with
MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
]

# Evaluation settings
EVAL_DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 1000  # Increased from 5 to 1000 for statistical significance


def run_baseline_evaluation(model_name: str, model_type: str, gpu_id: int, num_samples: int = 1000):
    """Run baseline evaluation on a specific GPU"""
    print(f"[GPU {gpu_id}] Starting baseline evaluation for {model_name}")
    
    # Create evaluation script
    eval_script = f"""
import sys
sys.path.append('src')
import torch
torch.cuda.set_device({gpu_id})

from src.medical_evaluator import MedicalEvaluator
from datetime import datetime
import json

evaluator = MedicalEvaluator("{model_name}", "cuda:{gpu_id}")
results = []

for dataset in {EVAL_DATASETS}:
    print(f"Evaluating on {{dataset}}...")
    result = evaluator.evaluate_dataset(dataset, {num_samples})
    result["model_type"] = "{model_type}"
    result["gpu_id"] = {gpu_id}
    result["timestamp"] = datetime.now().isoformat()
    results.append(result)

# Save results
output_file = "results/baseline_{model_name.replace('/', '_')}_{model_type}_gpu{gpu_id}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {{output_file}}")
"""
    
    # Write and execute script
    script_path = f"temp_eval_{gpu_id}.py"
    with open(script_path, 'w') as f:
        f.write(eval_script)
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        os.remove(script_path)
        return f"results/baseline_{model_name.replace('/', '_')}_{model_type}_gpu{gpu_id}.json"
    except Exception as e:
        print(f"[GPU {gpu_id}] Error evaluating {model_name}: {e}")
        if os.path.exists(script_path):
            os.remove(script_path)
        return None


def run_grpo_training(model_name: str, gpu_id: int, num_examples: int = 5000):
    """Run GRPO training on a specific GPU"""
    print(f"[GPU {gpu_id}] Starting GRPO training for {model_name}")
    
    cmd = [
        sys.executable,
        "run_gpu_grpo_training.py",
        "--model", model_name,
        "--gpu", str(gpu_id),
        "--num-examples", str(num_examples),
        "--batch-size", "16"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[GPU {gpu_id}] GRPO training completed for {model_name}")
        
        # Extract output path from stdout
        for line in result.stdout.split('\n'):
            if "Model saved to:" in line:
                model_path = line.split("Model saved to:")[-1].strip()
                return model_path
        
        return None
    except Exception as e:
        print(f"[GPU {gpu_id}] Error training {model_name}: {e}")
        return None


def combine_results(result_files: List[str], output_file: str):
    """Combine multiple result files into a single DataFrame"""
    all_results = []
    
    for file_path in result_files:
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"Combined results saved to {output_file}")
    return df


def generate_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Generate comprehensive comparison plots"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MedLogicTrace: Baseline vs GRPO Performance', fontsize=16)
    
    # 1. Accuracy comparison
    accuracy_data = df.groupby(['model', 'model_type', 'dataset'])['accuracy'].mean().reset_index()
    pivot_acc = accuracy_data.pivot_table(index=['model', 'dataset'], columns='model_type', values='accuracy')
    pivot_acc.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy by Model and Dataset')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(title='Model Type')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Token efficiency
    token_data = df.groupby(['model', 'model_type'])['avg_tokens'].mean().reset_index()
    token_data.plot(x='model', y='avg_tokens', hue='model_type', kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Average Token Usage')
    axes[0, 1].set_ylabel('Tokens')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy vs Token efficiency scatter
    axes[1, 0].scatter(df['avg_tokens'], df['accuracy'], c=df['model_type'].astype('category').cat.codes, 
                       alpha=0.6, s=100, cmap='viridis')
    axes[1, 0].set_xlabel('Average Tokens')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Token Efficiency')
    
    # Add legend
    for i, model_type in enumerate(df['model_type'].unique()):
        axes[1, 0].scatter([], [], c=plt.cm.viridis(i/len(df['model_type'].unique())), 
                          label=model_type, s=100)
    axes[1, 0].legend()
    
    # 4. Model size comparison
    model_sizes = {'0.5B': 0.5, '1.5B': 1.5, '1B': 1.0}
    df['model_size'] = df['model'].apply(lambda x: 
        0.5 if '0.5B' in x else 1.5 if '1.5B' in x else 1.0)
    
    size_perf = df.groupby(['model_size', 'model_type'])['accuracy'].mean().reset_index()
    pivot_size = size_perf.pivot(index='model_size', columns='model_type', values='accuracy')
    pivot_size.plot(kind='line', marker='o', markersize=10, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Model Size (B parameters)')
    axes[1, 1].set_ylabel('Average Accuracy')
    axes[1, 1].set_title('Performance Scaling with Model Size')
    axes[1, 1].legend(title='Model Type')
    
    plt.tight_layout()
    plot_path = f"{output_dir}/comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive comparison plot to {plot_path}")
    plt.close()
    
    # Generate summary statistics
    summary = df.groupby('model_type').agg({
        'accuracy': ['mean', 'std'],
        'avg_tokens': ['mean', 'std']
    }).round(3)
    
    summary_path = f"{output_dir}/summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")


def main():
    """Main experiment orchestrator"""
    print("="*80)
    print("MedLogicTrace Parallel GPU Experiments")
    print(f"Started at: {datetime.now()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/full_experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Phase 1: Baseline Evaluation (30-45 minutes)
    print("\nPHASE 1: Baseline Evaluation")
    print("-"*50)
    
    baseline_results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {}
        
        gpu_id = 0
        for base_model, instruct_model in MODELS:
            # Submit base model evaluation
            future = executor.submit(run_baseline_evaluation, base_model, "base", gpu_id, EVAL_SAMPLES)
            futures[future] = (base_model, "base")
            gpu_id += 1
            
            # Submit instruct model evaluation
            future = executor.submit(run_baseline_evaluation, instruct_model, "instruct", gpu_id, EVAL_SAMPLES)
            futures[future] = (instruct_model, "instruct")
            gpu_id += 1
        
        # Collect results
        for future in as_completed(futures):
            model_info = futures[future]
            result_file = future.result()
            if result_file:
                baseline_results.append(result_file)
                print(f"✓ Completed baseline evaluation for {model_info[0]} ({model_info[1]})")
    
    # Combine baseline results
    baseline_df = combine_results(baseline_results, f"{experiment_dir}/baseline_results.csv")
    
    # Phase 2: GRPO Training (1-1.5 hours)
    print("\nPHASE 2: GRPO Training")
    print("-"*50)
    
    grpo_models = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        for i, (_, instruct_model) in enumerate(MODELS):
            # Use GPU pairs for training (0-1, 2-3, 4-5)
            gpu_id = i * 2
            future = executor.submit(run_grpo_training, instruct_model, gpu_id, 5000)
            futures[future] = instruct_model
        
        # Collect trained models
        for future in as_completed(futures):
            model_name = futures[future]
            model_path = future.result()
            if model_path:
                grpo_models.append((model_path, "grpo"))
                print(f"✓ Completed GRPO training for {model_name}")
    
    # Phase 3: Post-GRPO Evaluation (30-45 minutes)
    print("\nPHASE 3: Post-GRPO Evaluation")
    print("-"*50)
    
    grpo_results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        for i, (model_path, model_type) in enumerate(grpo_models):
            future = executor.submit(run_baseline_evaluation, model_path, model_type, i, EVAL_SAMPLES)
            futures[future] = model_path
        
        # Collect results
        for future in as_completed(futures):
            model_path = futures[future]
            result_file = future.result()
            if result_file:
                grpo_results.append(result_file)
                print(f"✓ Completed evaluation for GRPO model: {model_path}")
    
    # Combine all results
    all_result_files = baseline_results + grpo_results
    final_df = combine_results(all_result_files, f"{experiment_dir}/all_results.csv")
    
    # Phase 4: Generate plots and analysis
    print("\nPHASE 4: Analysis and Visualization")
    print("-"*50)
    
    generate_comparison_plots(final_df, experiment_dir)
    
    # Save experiment metadata
    metadata = {
        "start_time": timestamp,
        "end_time": datetime.now().isoformat(),
        "models_tested": MODELS,
        "eval_datasets": EVAL_DATASETS,
        "eval_samples": EVAL_SAMPLES,
        "grpo_training_samples": 5000,
        "num_gpus_used": torch.cuda.device_count(),
        "results_directory": experiment_dir
    }
    
    with open(f"{experiment_dir}/experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results saved to: {experiment_dir}")
    print(f"Total time: {(datetime.now() - datetime.strptime(timestamp, '%Y%m%d_%H%M%S')).total_seconds() / 3600:.1f} hours")
    
    # Print key findings
    if not final_df.empty:
        print("\nKey Findings:")
        summary = final_df.groupby('model_type')[['accuracy', 'avg_tokens']].mean()
        print(summary)


if __name__ == "__main__":
    main()

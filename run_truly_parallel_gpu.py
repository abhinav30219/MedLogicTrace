#!/usr/bin/env python3
"""Truly parallel GPU experiments with proper device handling"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
import json
import time
import gc
from typing import List, Tuple, Dict

# Use only open-access models
MODELS_TO_TEST = [
    ("Qwen/Qwen2.5-0.5B", "base"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct"),
    ("Qwen/Qwen2.5-1.5B", "base"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "instruct"),
    # Adding more Qwen variants to utilize all GPUs
    ("Qwen/Qwen2.5-Coder-1.5B", "coder"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "coder-instruct"),
]

DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 500


def evaluate_single_model_on_gpu(model_name: str, model_type: str, gpu_id: int, 
                                 datasets: List[str], num_samples: int) -> List[Dict]:
    """
    Evaluate a single model on a specific GPU.
    This function runs in a separate process.
    """
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # Import inside function to avoid pickling issues
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    import gc
    
    print(f"[GPU {gpu_id}] Starting evaluation of {model_name} ({model_type})")
    
    results = []
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model directly to specific GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )
        model.eval()
        
        # Evaluate on each dataset
        for dataset_name in datasets:
            print(f"[GPU {gpu_id}] Evaluating {model_name} on {dataset_name}")
            
            # Import data utils
            from src.data_utils import load_medical_dataset, format_medical_prompt
            
            examples = load_medical_dataset(dataset_name, num_samples)
            
            correct = 0
            total = 0
            token_counts = []
            
            # Process examples
            for i, example in enumerate(examples):
                if i % 100 == 0:
                    print(f"[GPU {gpu_id}] {model_name}: {i}/{len(examples)} examples")
                
                prompt = format_medical_prompt(example)
                
                # Generate response
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                with torch.no_grad():
                    try:
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=128,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                        
                        # Check accuracy
                        if "options" in example and response:
                            answer_letter = chr(65 + example["answer"])
                            if answer_letter.lower() in response.lower()[:50]:
                                correct += 1
                            
                            token_counts.append(len(tokenizer.encode(response)))
                        
                        total += 1
                        
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Generation error: {e}")
                        total += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            
            result = {
                "model": model_name,
                "model_type": model_type,
                "dataset": dataset_name,
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "total_examples": total,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"[GPU {gpu_id}] {model_name} on {dataset_name}: {accuracy:.2%} accuracy, {avg_tokens:.1f} tokens")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[GPU {gpu_id}] ✓ Completed {model_name}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error with {model_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def train_grpo_on_gpu(model_name: str, gpu_id: int, num_examples: int = 2000) -> str:
    """
    Train a model with GRPO on a specific GPU.
    This function runs in a separate process.
    """
    import torch
    torch.cuda.set_device(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting GRPO training for {model_name}")
    
    # Import the GPU GRPO training script
    import subprocess
    
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
        
        # Extract output path
        for line in result.stdout.split('\n'):
            if "Model saved to:" in line:
                model_path = line.split("Model saved to:")[-1].strip()
                print(f"[GPU {gpu_id}] ✓ GRPO training completed for {model_name}")
                return model_path
        
        return f"{model_name}_grpo"
        
    except Exception as e:
        print(f"[GPU {gpu_id}] GRPO training error for {model_name}: {e}")
        return None


def main():
    print("="*80)
    print("MedLogicTrace Truly Parallel GPU Experiments")
    print(f"Started: {datetime.now()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/parallel_gpu_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # PHASE 1: Parallel Baseline Evaluation
    print("\nPHASE 1: Parallel Baseline Evaluation")
    print("-"*50)
    print(f"Running {len(MODELS_TO_TEST)} models in parallel across {torch.cuda.device_count()} GPUs")
    
    start_time = time.time()
    all_results = []
    
    with ProcessPoolExecutor(max_workers=min(len(MODELS_TO_TEST), torch.cuda.device_count())) as executor:
        # Submit all evaluation jobs
        futures = {}
        for i, (model_name, model_type) in enumerate(MODELS_TO_TEST):
            gpu_id = i % torch.cuda.device_count()
            
            future = executor.submit(
                evaluate_single_model_on_gpu,
                model_name, model_type, gpu_id,
                DATASETS, EVAL_SAMPLES
            )
            futures[future] = (model_name, model_type, gpu_id)
        
        # Collect results as they complete
        for future in as_completed(futures):
            model_info = futures[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
                    print(f"✓ Received results for {model_info[0]} from GPU {model_info[2]}")
            except Exception as e:
                print(f"✗ Error getting results for {model_info[0]}: {e}")
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 completed in {phase1_time/60:.1f} minutes")
    
    # Save baseline results
    if all_results:
        baseline_df = pd.DataFrame(all_results)
        baseline_path = f"{results_dir}/baseline_results.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"Baseline results saved to {baseline_path}")
        
        # Show summary
        print("\nBaseline Summary:")
        summary = baseline_df.groupby('model_type')[['accuracy', 'avg_tokens']].mean().round(3)
        print(summary)
    
    # PHASE 2: Parallel GRPO Training (for instruct models)
    print("\nPHASE 2: Parallel GRPO Training")
    print("-"*50)
    
    instruct_models = [(m, t) for m, t in MODELS_TO_TEST if 'instruct' in t.lower()]
    print(f"Training {len(instruct_models)} instruction models with GRPO")
    
    start_time = time.time()
    grpo_models = []
    
    with ProcessPoolExecutor(max_workers=min(len(instruct_models), torch.cuda.device_count()//2)) as executor:
        # Submit GRPO training jobs (using GPU pairs for more memory)
        futures = {}
        for i, (model_name, model_type) in enumerate(instruct_models):
            gpu_id = i * 2  # Use even GPUs (0, 2, 4, 6)
            
            future = executor.submit(
                train_grpo_on_gpu,
                model_name, gpu_id, 2000
            )
            futures[future] = (model_name, gpu_id)
        
        # Collect trained models
        for future in as_completed(futures):
            model_info = futures[future]
            try:
                model_path = future.result()
                if model_path:
                    grpo_models.append((model_path, "grpo"))
                    print(f"✓ GRPO training completed for {model_info[0]}")
            except Exception as e:
                print(f"✗ GRPO training failed for {model_info[0]}: {e}")
    
    phase2_time = time.time() - start_time
    print(f"\nPhase 2 completed in {phase2_time/60:.1f} minutes")
    
    # PHASE 3: Evaluate GRPO models
    if grpo_models:
        print("\nPHASE 3: Parallel GRPO Model Evaluation")
        print("-"*50)
        
        start_time = time.time()
        grpo_results = []
        
        with ProcessPoolExecutor(max_workers=len(grpo_models)) as executor:
            futures = {}
            for i, (model_path, model_type) in enumerate(grpo_models):
                gpu_id = i % torch.cuda.device_count()
                
                future = executor.submit(
                    evaluate_single_model_on_gpu,
                    model_path, model_type, gpu_id,
                    DATASETS, EVAL_SAMPLES
                )
                futures[future] = (model_path, gpu_id)
            
            for future in as_completed(futures):
                model_info = futures[future]
                try:
                    results = future.result()
                    if results:
                        grpo_results.extend(results)
                        print(f"✓ Evaluation completed for GRPO model on GPU {model_info[1]}")
                except Exception as e:
                    print(f"✗ Evaluation failed for GRPO model: {e}")
        
        phase3_time = time.time() - start_time
        print(f"\nPhase 3 completed in {phase3_time/60:.1f} minutes")
        
        # Combine all results
        all_results.extend(grpo_results)
    
    # Save final results and generate plots
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_path = f"{results_dir}/all_results.csv"
        final_df.to_csv(final_path, index=False)
        
        # Generate plots
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('MedLogicTrace Parallel GPU Results', fontsize=16)
            
            # 1. Accuracy by model type
            model_acc = final_df.groupby('model_type')['accuracy'].mean().sort_values(ascending=False)
            model_acc.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Accuracy by Model Type')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # 2. Token efficiency
            model_tokens = final_df.groupby('model_type')['avg_tokens'].mean().sort_values()
            model_tokens.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Average Token Usage by Model Type')
            ax2.set_ylabel('Tokens')
            
            # 3. Dataset comparison
            dataset_acc = final_df.pivot_table(index='dataset', columns='model_type', values='accuracy')
            dataset_acc.plot(kind='bar', ax=ax3)
            ax3.set_title('Accuracy by Dataset and Model Type')
            ax3.set_ylabel('Accuracy')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 4. Efficiency vs Accuracy scatter
            ax4.scatter(final_df['avg_tokens'], final_df['accuracy'], 
                       c=final_df['model_type'].astype('category').cat.codes, 
                       alpha=0.6, s=100, cmap='viridis')
            ax4.set_xlabel('Average Tokens')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Token Efficiency vs Accuracy Trade-off')
            
            plt.tight_layout()
            plot_path = f"{results_dir}/parallel_results_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate plots: {e}")
    
    # Final summary
    total_time = phase1_time + phase2_time + (phase3_time if grpo_models else 0)
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Total experiment time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    
    if all_results:
        print("\nFinal Summary:")
        final_summary = final_df.groupby('model_type')[['accuracy', 'avg_tokens']].mean().round(3)
        print(final_summary)
        
        # Calculate speedup
        sequential_estimate = len(MODELS_TO_TEST) * len(DATASETS) * EVAL_SAMPLES * 0.5 / 60  # ~0.5s per example
        print(f"\nEstimated sequential time: {sequential_estimate:.1f} minutes")
        print(f"Actual parallel time: {total_time/60:.1f} minutes")
        print(f"Speedup: {sequential_estimate/(total_time/60):.1f}x")


if __name__ == "__main__":
    main()

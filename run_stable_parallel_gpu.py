#!/usr/bin/env python3
"""Stable parallel GPU experiments with numerical stability fixes"""

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
    # Skip larger models that have stability issues
    # ("Qwen/Qwen2.5-1.5B", "base"),
    # ("Qwen/Qwen2.5-1.5B-Instruct", "instruct"),
]

DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 100  # Reduced for testing stability


def evaluate_single_model_on_gpu(model_name: str, model_type: str, gpu_id: int, 
                                 datasets: List[str], num_samples: int) -> List[Dict]:
    """
    Evaluate a single model on a specific GPU with numerical stability fixes.
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
        
        # Load model with float32 for stability
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for numerical stability
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
            generation_failures = 0
            
            # Process examples
            for i, example in enumerate(examples):
                if i % 20 == 0:
                    print(f"[GPU {gpu_id}] {model_name}: {i}/{len(examples)} examples (failures: {generation_failures})")
                
                prompt = format_medical_prompt(example)
                
                # Generate response with safer parameters
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                with torch.no_grad():
                    try:
                        # Use safer generation parameters
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50,  # Reduced for stability
                            temperature=0.8,    # Slightly higher for stability
                            do_sample=True,
                            top_p=0.95,        # Higher top_p for stability
                            top_k=50,          # Add top_k for additional stability
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,  # Prevent repetition
                        )
                        
                        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                        
                        # Check accuracy
                        if "options" in example and response:
                            answer_letter = chr(65 + example["answer"])
                            if answer_letter.lower() in response.lower()[:50]:
                                correct += 1
                            
                            token_counts.append(len(tokenizer.encode(response)))
                        
                        total += 1
                        
                    except RuntimeError as e:
                        if "inf" in str(e) or "nan" in str(e):
                            generation_failures += 1
                            # Try with greedy decoding as fallback
                            try:
                                outputs = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=20,
                                    do_sample=False,  # Greedy decoding
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
                            except:
                                total += 1
                        else:
                            print(f"[GPU {gpu_id}] Unexpected error: {e}")
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
                "generation_failures": generation_failures,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            print(f"[GPU {gpu_id}] {model_name} on {dataset_name}: {accuracy:.2%} accuracy, {avg_tokens:.1f} tokens, {generation_failures} failures")
        
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


def main():
    print("="*80)
    print("MedLogicTrace Stable Parallel GPU Experiments")
    print(f"Started: {datetime.now()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/stable_parallel_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # PHASE 1: Parallel Baseline Evaluation
    print("\nPHASE 1: Parallel Baseline Evaluation (with stability fixes)")
    print("-"*50)
    print(f"Running {len(MODELS_TO_TEST)} models in parallel across GPUs")
    print("Using float32 precision and safer generation parameters for stability")
    
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
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = f"{results_dir}/evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Generate summary
        print("\nSUMMARY:")
        print("-"*50)
        summary = df.groupby('model_type')[['accuracy', 'avg_tokens', 'generation_failures']].mean().round(3)
        print(summary)
        
        # Simple plot
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Accuracy by model
            model_acc = df.groupby(['model', 'model_type'])['accuracy'].mean().reset_index()
            model_acc.pivot(index='model', columns='model_type', values='accuracy').plot(kind='bar', ax=ax1)
            ax1.set_title('Accuracy by Model')
            ax1.set_ylabel('Accuracy')
            ax1.tick_params(axis='x', rotation=45)
            
            # Token efficiency
            model_tokens = df.groupby(['model', 'model_type'])['avg_tokens'].mean().reset_index()
            model_tokens.pivot(index='model', columns='model_type', values='avg_tokens').plot(kind='bar', ax=ax2)
            ax2.set_title('Average Tokens by Model')
            ax2.set_ylabel('Tokens')
            ax2.tick_params(axis='x', rotation=45)
            
            # Generation failures
            model_failures = df.groupby(['model', 'model_type'])['generation_failures'].mean().reset_index()
            model_failures.pivot(index='model', columns='model_type', values='generation_failures').plot(kind='bar', ax=ax3)
            ax3.set_title('Generation Failures by Model')
            ax3.set_ylabel('Number of Failures')
            ax3.tick_params(axis='x', rotation=45)
            
            # Dataset comparison
            dataset_acc = df.pivot_table(index='dataset', columns='model_type', values='accuracy')
            dataset_acc.plot(kind='bar', ax=ax4)
            ax4.set_title('Accuracy by Dataset')
            ax4.set_ylabel('Accuracy')
            
            plt.tight_layout()
            plot_path = f"{results_dir}/comparison_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    print(f"\nExperiment completed at {datetime.now()}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Print key insights
    if all_results:
        print("\nKey Insights:")
        print(f"- Models evaluated: {len(MODELS_TO_TEST)}")
        print(f"- Total evaluations: {len(all_results)}")
        print(f"- Average accuracy: {df['accuracy'].mean():.2%}")
        print(f"- Average tokens per response: {df['avg_tokens'].mean():.1f}")
        print(f"- Average generation failures per run: {df['generation_failures'].mean():.1f}")


if __name__ == "__main__":
    main()

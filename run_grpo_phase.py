#!/usr/bin/env python3
"""Run just the GRPO phases of the experiment"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import time
import gc
from typing import List, Dict

# Models for GRPO training
INSTRUCT_MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct-0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "instruct-1.5B"),
]

DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 500


def train_grpo_on_gpu(model_name: str, model_type: str, gpu_id: int, num_examples: int = 500) -> Dict:
    """
    Train a model with GRPO on a specific GPU using simplified approach.
    """
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"[GPU {gpu_id}] Starting GRPO training for {model_name}")
    
    try:
        # Run the existing GRPO training script directly
        import subprocess
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/grpo_{model_name.split('/')[-1]}_{timestamp}"
        
        # Use the existing run_gpu_grpo_training.py script
        cmd = [
            sys.executable,
            "run_gpu_grpo_training.py",
            "--model-name", model_name,
            "--num-examples", str(num_examples),
            "--output-dir", output_dir,
            "--device", f"cuda:{gpu_id}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ GRPO training completed for {model_name}")
            return {
                "model_path": output_dir,
                "original_model": model_name,
                "model_type": model_type,
                "gpu_id": gpu_id
            }
        else:
            print(f"[GPU {gpu_id}] GRPO training failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"[GPU {gpu_id}] GRPO training error for {model_name}: {e}")
        return None


def evaluate_model_on_gpu(model_path: str, model_type: str, gpu_id: int) -> List[Dict]:
    """Evaluate a GRPO-trained model"""
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data_utils import load_medical_dataset, format_medical_prompt
    
    print(f"[GPU {gpu_id}] Evaluating {model_path}")
    
    results = []
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dtype = torch.float32 if "1.5B" in model_path else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True
        )
        model.eval()
        
        # Evaluate on each dataset
        for dataset_name in DATASETS:
            examples = load_medical_dataset(dataset_name, EVAL_SAMPLES)
            
            correct = 0
            total = 0
            token_counts = []
            
            for i, example in enumerate(examples):
                if i % 100 == 0:
                    print(f"[GPU {gpu_id}] Progress: {i}/{len(examples)}")
                
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
            
            accuracy = correct / total if total > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            
            results.append({
                "model": model_path,
                "model_type": model_type,
                "dataset": dataset_name,
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "total_examples": total,
                "gpu_id": gpu_id,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"[GPU {gpu_id}] {dataset_name}: {accuracy:.2%} accuracy, {avg_tokens:.1f} tokens")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error evaluating {model_path}: {e}")
    
    return results


def main():
    print("="*80)
    print("MedLogicTrace GRPO Training and Evaluation")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/grpo_experiments_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # PHASE 1: GRPO Training
    print("\nPHASE 1: GRPO Training")
    print("-"*50)
    
    start_time = time.time()
    grpo_models = []
    
    # Run GRPO training sequentially to avoid memory issues
    for i, (model_name, model_type) in enumerate(INSTRUCT_MODELS):
        gpu_id = i * 2  # Use GPUs 0 and 2
        result = train_grpo_on_gpu(model_name, model_type, gpu_id, 500)
        if result:
            grpo_models.append(result)
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 completed in {phase1_time/60:.1f} minutes")
    print(f"Successfully trained {len(grpo_models)} models")
    
    # PHASE 2: Evaluate GRPO Models
    if grpo_models:
        print("\nPHASE 2: GRPO Model Evaluation")
        print("-"*50)
        
        start_time = time.time()
        all_results = []
        
        with ProcessPoolExecutor(max_workers=len(grpo_models)) as executor:
            futures = {}
            for i, model_info in enumerate(grpo_models):
                gpu_id = i % torch.cuda.device_count()
                future = executor.submit(
                    evaluate_model_on_gpu,
                    model_info["model_path"],
                    f"{model_info['model_type']}-grpo",
                    gpu_id
                )
                futures[future] = model_info
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                        print(f"✓ Evaluation completed for {futures[future]['model_path']}")
                except Exception as e:
                    print(f"✗ Evaluation failed: {e}")
        
        phase2_time = time.time() - start_time
        print(f"\nPhase 2 completed in {phase2_time/60:.1f} minutes")
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = f"{results_dir}/grpo_results.csv"
            results_df.to_csv(results_path, index=False)
            
            print("\nGRPO RESULTS SUMMARY:")
            print("-"*50)
            summary = results_df.groupby('model_type')[['accuracy', 'avg_tokens']].mean().round(3)
            print(summary)
            
            # Load previous baseline results for comparison
            try:
                # Find most recent baseline results
                import glob
                baseline_files = glob.glob("results/full_experiments_*/baseline_results.csv")
                if baseline_files:
                    latest_baseline = sorted(baseline_files)[-1]
                    baseline_df = pd.read_csv(latest_baseline)
                    
                    print("\nCOMPARISON WITH BASELINE:")
                    print("-"*50)
                    
                    for model_type in ["instruct-0.5B", "instruct-1.5B"]:
                        baseline_acc = baseline_df[baseline_df['model_type'] == model_type]['accuracy'].mean()
                        grpo_acc = results_df[results_df['model_type'] == f"{model_type}-grpo"]['accuracy'].mean()
                        
                        print(f"{model_type}:")
                        print(f"  Baseline: {baseline_acc:.2%}")
                        print(f"  GRPO:     {grpo_acc:.2%}")
                        print(f"  Change:   {(grpo_acc - baseline_acc):.2%}")
            except Exception as e:
                print(f"Could not load baseline results for comparison: {e}")
    
    total_time = phase1_time + (phase2_time if grpo_models else 0)
    print("\n" + "="*80)
    print("GRPO EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

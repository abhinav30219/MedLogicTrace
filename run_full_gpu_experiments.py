#!/usr/bin/env python3
"""Full GPU experiments with 500 examples, multiple models, and GRPO training"""

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

# Full model suite for experiments
MODELS_TO_TEST = [
    ("Qwen/Qwen2.5-0.5B", "base-0.5B"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct-0.5B"),
    ("Qwen/Qwen2.5-1.5B", "base-1.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "instruct-1.5B"),
]

DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 500  # Full evaluation


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
        
        # Determine dtype based on model size (larger models need float32 for stability)
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        
        # Load model with appropriate precision
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
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
                if i % 50 == 0:
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
                            max_new_tokens=64,   # Reasonable for medical Q&A
                            temperature=0.8,     # Stability
                            do_sample=True,
                            top_p=0.95,         # Higher for stability
                            top_k=50,           # Additional stability
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            repetition_penalty=1.1,
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
                                    do_sample=False,  # Greedy
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


def train_grpo_on_gpu(model_name: str, gpu_id: int, num_examples: int = 1000) -> Dict:
    """
    Train a model with GRPO on a specific GPU.
    Using TRL's GRPOTrainer directly for compatibility.
    """
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"[GPU {gpu_id}] Starting GRPO training for {model_name}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        from src.data_utils import load_math_dataset
        import gc
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate precision
        dtype = torch.float32 if "1.5B" in model_name else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        # Load training data
        train_data = load_math_dataset("gsm8k", num_examples)
        
        # Configure GRPO
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/grpo_{model_name.split('/')[-1]}_{timestamp}"
        
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=4,  # Small batch for stability
            gradient_accumulation_steps=4,   # Effective batch size = 16
            learning_rate=1e-5,
            num_train_epochs=1,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            
            # GRPO specific
            kl_coef=0.05,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.95,
            
            # Memory optimization
            gradient_checkpointing=True,
            fp16=False,  # Use dtype from model loading
            
            # Disable for GPU
            dataloader_pin_memory=True,
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            config=grpo_config,
            tokenizer=tokenizer,
            train_dataset=train_data,
        )
        
        # Train
        print(f"[GPU {gpu_id}] Training {model_name} with GRPO...")
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"[GPU {gpu_id}] ✓ GRPO training completed for {model_name}")
        
        return {
            "model_path": output_dir,
            "original_model": model_name,
            "gpu_id": gpu_id
        }
        
    except Exception as e:
        print(f"[GPU {gpu_id}] GRPO training error for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("MedLogicTrace Full GPU Experiments (500 examples)")
    print(f"Started: {datetime.now()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/full_experiments_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # PHASE 1: Parallel Baseline Evaluation (500 examples)
    print("\nPHASE 1: Full Baseline Evaluation (500 examples)")
    print("-"*50)
    print(f"Running {len(MODELS_TO_TEST)} models across {min(len(MODELS_TO_TEST), torch.cuda.device_count())} GPUs")
    
    start_time = time.time()
    baseline_results = []
    
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
                    baseline_results.extend(results)
                    print(f"✓ Received baseline results for {model_info[0]}")
            except Exception as e:
                print(f"✗ Error getting baseline results for {model_info[0]}: {e}")
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 completed in {phase1_time/60:.1f} minutes")
    
    # Save baseline results
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        baseline_path = f"{results_dir}/baseline_results.csv"
        baseline_df.to_csv(baseline_path, index=False)
        print(f"\nBaseline results saved to {baseline_path}")
        
        # Show baseline summary
        print("\nBASELINE SUMMARY:")
        print("-"*50)
        summary = baseline_df.groupby('model_type')[['accuracy', 'avg_tokens', 'generation_failures']].mean().round(3)
        print(summary)
    
    # PHASE 2: GRPO Training for Instruct Models
    print("\nPHASE 2: GRPO Training")
    print("-"*50)
    
    # Select instruct models for GRPO training
    instruct_models = [(m, t) for m, t in MODELS_TO_TEST if 'instruct' in t.lower()]
    print(f"Training {len(instruct_models)} instruction-tuned models with GRPO")
    
    start_time = time.time()
    grpo_models = []
    
    with ProcessPoolExecutor(max_workers=min(len(instruct_models), torch.cuda.device_count()//2)) as executor:
        # Submit GRPO training jobs (using every other GPU for memory)
        futures = {}
        for i, (model_name, model_type) in enumerate(instruct_models):
            gpu_id = (i * 2) % torch.cuda.device_count()
            
            future = executor.submit(
                train_grpo_on_gpu,
                model_name, gpu_id, 1000  # Reduced for faster experiments
            )
            futures[future] = (model_name, model_type, gpu_id)
        
        # Collect trained models
        for future in as_completed(futures):
            model_info = futures[future]
            try:
                result = future.result()
                if result:
                    grpo_models.append((result["model_path"], f"{model_info[1]}-grpo"))
                    print(f"✓ GRPO training completed for {model_info[0]}")
            except Exception as e:
                print(f"✗ GRPO training failed for {model_info[0]}: {e}")
    
    phase2_time = time.time() - start_time
    print(f"\nPhase 2 completed in {phase2_time/60:.1f} minutes")
    
    # PHASE 3: Evaluate GRPO Models
    if grpo_models:
        print("\nPHASE 3: GRPO Model Evaluation")
        print("-"*50)
        print(f"Evaluating {len(grpo_models)} GRPO-trained models")
        
        start_time = time.time()
        grpo_results = []
        
        with ProcessPoolExecutor(max_workers=min(len(grpo_models), torch.cuda.device_count())) as executor:
            futures = {}
            for i, (model_path, model_type) in enumerate(grpo_models):
                gpu_id = i % torch.cuda.device_count()
                
                future = executor.submit(
                    evaluate_single_model_on_gpu,
                    model_path, model_type, gpu_id,
                    DATASETS, EVAL_SAMPLES
                )
                futures[future] = (model_path, model_type, gpu_id)
            
            for future in as_completed(futures):
                model_info = futures[future]
                try:
                    results = future.result()
                    if results:
                        grpo_results.extend(results)
                        print(f"✓ Evaluation completed for {model_info[0]}")
                except Exception as e:
                    print(f"✗ Evaluation failed for {model_info[0]}: {e}")
        
        phase3_time = time.time() - start_time
        print(f"\nPhase 3 completed in {phase3_time/60:.1f} minutes")
    else:
        grpo_results = []
        phase3_time = 0
    
    # Combine all results
    all_results = baseline_results + grpo_results
    
    # Save final results and generate comprehensive plots
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_path = f"{results_dir}/all_results.csv"
        final_df.to_csv(final_path, index=False)
        
        print("\nFINAL SUMMARY:")
        print("-"*50)
        final_summary = final_df.groupby('model_type')[['accuracy', 'avg_tokens', 'generation_failures']].mean().round(3)
        print(final_summary)
        
        # Generate comprehensive plots
        try:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(16, 12))
            
            # 1. Accuracy comparison
            ax1 = plt.subplot(2, 3, 1)
            model_acc = final_df.groupby('model_type')['accuracy'].mean().sort_values(ascending=False)
            model_acc.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Average Accuracy by Model Type', fontsize=14)
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Token efficiency
            ax2 = plt.subplot(2, 3, 2)
            model_tokens = final_df.groupby('model_type')['avg_tokens'].mean().sort_values()
            model_tokens.plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Average Token Usage by Model Type', fontsize=14)
            ax2.set_ylabel('Tokens')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Dataset comparison
            ax3 = plt.subplot(2, 3, 3)
            dataset_acc = final_df.pivot_table(index='dataset', columns='model_type', values='accuracy')
            dataset_acc.plot(kind='bar', ax=ax3)
            ax3.set_title('Accuracy by Dataset and Model Type', fontsize=14)
            ax3.set_ylabel('Accuracy')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 4. Efficiency vs Accuracy scatter
            ax4 = plt.subplot(2, 3, 4)
            colors = plt.cm.tab10(np.linspace(0, 1, len(final_df['model_type'].unique())))
            for i, model_type in enumerate(final_df['model_type'].unique()):
                mask = final_df['model_type'] == model_type
                ax4.scatter(final_df[mask]['avg_tokens'], final_df[mask]['accuracy'], 
                           label=model_type, alpha=0.7, s=100, c=[colors[i]])
            ax4.set_xlabel('Average Tokens')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Token Efficiency vs Accuracy Trade-off', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. GRPO vs Baseline comparison (if GRPO models exist)
            if grpo_results:
                ax5 = plt.subplot(2, 3, 5)
                # Compare instruct vs instruct-grpo
                comparison_data = []
                for model_type in ['instruct-0.5B', 'instruct-1.5B']:
                    baseline_acc = final_df[final_df['model_type'] == model_type]['accuracy'].mean()
                    grpo_acc = final_df[final_df['model_type'] == f"{model_type}-grpo"]['accuracy'].mean()
                    if not pd.isna(baseline_acc) and not pd.isna(grpo_acc):
                        comparison_data.append({
                            'Model': model_type.replace('instruct-', ''),
                            'Baseline': baseline_acc,
                            'GRPO': grpo_acc
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df.set_index('Model').plot(kind='bar', ax=ax5)
                    ax5.set_title('GRPO Impact on Accuracy', fontsize=14)
                    ax5.set_ylabel('Accuracy')
                    ax5.set_ylim(0, 1)
            
            # 6. Generation failures
            ax6 = plt.subplot(2, 3, 6)
            model_failures = final_df.groupby('model_type')['generation_failures'].mean().sort_values(ascending=False)
            model_failures.plot(kind='bar', ax=ax6, color='salmon')
            ax6.set_title('Average Generation Failures by Model Type', fontsize=14)
            ax6.set_ylabel('Number of Failures')
            ax6.tick_params(axis='x', rotation=45)
            
            plt.suptitle('MedLogicTrace Full Experiment Results (500 Examples)', fontsize=16)
            plt.tight_layout()
            plot_path = f"{results_dir}/full_results_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nComprehensive plot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate plots: {e}")
    
    # Final timing summary
    total_time = phase1_time + phase2_time + phase3_time
    print("\n" + "="*80)
    print("FULL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"Total experiment time: {total_time/60:.1f} minutes")
    print(f"- Phase 1 (Baseline): {phase1_time/60:.1f} minutes")
    print(f"- Phase 2 (GRPO Training): {phase2_time/60:.1f} minutes")
    print(f"- Phase 3 (GRPO Evaluation): {phase3_time/60:.1f} minutes")
    print(f"\nResults saved to: {results_dir}")
    
    # Key insights
    if all_results:
        print("\nKEY INSIGHTS:")
        print(f"- Models evaluated: {len(MODELS_TO_TEST)} baseline + {len(grpo_models)} GRPO")
        print(f"- Total evaluations: {len(all_results)}")
        print(f"- Average baseline accuracy: {baseline_df['accuracy'].mean():.2%}")
        if grpo_results:
            grpo_df = pd.DataFrame(grpo_results)
            print(f"- Average GRPO accuracy: {grpo_df['accuracy'].mean():.2%}")
            print(f"- GRPO improvement: {(grpo_df['accuracy'].mean() - baseline_df[baseline_df['model_type'].str.contains('instruct')]['accuracy'].mean()):.2%}")


if __name__ == "__main__":
    main()

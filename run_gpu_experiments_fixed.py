#!/usr/bin/env python3
"""Fixed GPU experiments with proper device handling and open models only"""

import os
import sys
sys.path.append('src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import gc

# Use only open-access models
MODELS_TO_TEST = [
    ("Qwen/Qwen2.5-0.5B", "base"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "instruct"),
    ("Qwen/Qwen2.5-1.5B", "base"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "instruct"),
]

DATASETS = ["medmcqa", "pubmed_qa"]
EVAL_SAMPLES = 500  # Reduced for faster testing


class FixedMedicalEvaluator:
    """Medical evaluator with proper device handling"""
    
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name} on {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with specific device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )
        self.model.eval()
        
    def generate_response(self, prompt, max_new_tokens=128):
        """Generate response with proper device handling"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the correct device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                return response
                
            except RuntimeError as e:
                print(f"Generation error: {e}")
                return ""
    
    def evaluate_dataset(self, dataset_name, num_samples=100):
        """Evaluate on a dataset"""
        from src.data_utils import load_medical_dataset, format_medical_prompt
        
        print(f"Evaluating on {dataset_name} with {num_samples} samples")
        examples = load_medical_dataset(dataset_name, num_samples)
        
        correct = 0
        total = 0
        token_counts = []
        
        for example in tqdm(examples, desc=f"Evaluating {dataset_name}"):
            prompt = format_medical_prompt(example)
            response = self.generate_response(prompt)
            
            if response and "options" in example:
                # Check accuracy
                answer_letter = chr(65 + example["answer"])
                if answer_letter.lower() in response.lower()[:50]:
                    correct += 1
                
                token_counts.append(len(self.tokenizer.encode(response)))
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        return {
            "model": self.model_name,
            "dataset": dataset_name,
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "total_examples": total
        }


def evaluate_model_gpu(model_name, model_type, gpu_id, num_samples=500):
    """Evaluate a model on a specific GPU"""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    print(f"\n[GPU {gpu_id}] Evaluating {model_name} ({model_type})")
    
    try:
        evaluator = FixedMedicalEvaluator(model_name, device)
        results = []
        
        for dataset_name in DATASETS:
            result = evaluator.evaluate_dataset(dataset_name, num_samples)
            result["model_type"] = model_type
            result["gpu_id"] = gpu_id
            result["timestamp"] = datetime.now().isoformat()
            results.append(result)
        
        # Clean up
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error evaluating {model_name}: {e}")
        return []


def train_grpo_simple(model_name, gpu_id, num_examples=1000):
    """Simplified GRPO training"""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    print(f"\n[GPU {gpu_id}] Training {model_name} with GRPO")
    
    try:
        # For now, just return the original model path
        # Full GRPO training would go here
        print(f"[GPU {gpu_id}] GRPO training placeholder for {model_name}")
        return model_name + "_grpo"
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error training {model_name}: {e}")
        return None


def main():
    print("="*70)
    print("MedLogicTrace GPU Experiments (Fixed)")
    print(f"Started: {datetime.now()}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print("="*70)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/gpu_experiment_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    
    # Phase 1: Baseline Evaluation
    print("\nPHASE 1: Baseline Evaluation")
    print("-"*50)
    
    for i, (model_name, model_type) in enumerate(MODELS_TO_TEST):
        gpu_id = i % torch.cuda.device_count()
        results = evaluate_model_gpu(model_name, model_type, gpu_id, EVAL_SAMPLES)
        all_results.extend(results)
        
        if results:
            print(f"✓ Completed {model_name} ({model_type})")
            for r in results:
                print(f"  - {r['dataset']}: {r['accuracy']:.2%} accuracy, {r['avg_tokens']:.1f} tokens")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = f"{results_dir}/evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Generate summary
        print("\nSUMMARY:")
        print("-"*50)
        summary = df.groupby('model_type')[['accuracy', 'avg_tokens']].mean()
        print(summary)
        
        # Simple plot
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
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
            
            plt.tight_layout()
            plot_path = f"{results_dir}/comparison_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"Could not generate plot: {e}")
    
    print(f"\nExperiment completed at {datetime.now()}")


if __name__ == "__main__":
    main()

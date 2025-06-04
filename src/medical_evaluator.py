"""Medical dataset evaluation utilities"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from .data_utils import (
    load_medical_dataset,
    format_medical_prompt,
    evaluate_medical_response,
    calculate_token_efficiency
)


class MedicalEvaluator:
    """Evaluator for medical reasoning tasks"""
    
    def __init__(self, model_path: str, device: str = "mps"):
        self.model_path = model_path
        self.device = device
        
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": device} if device == "mps" else "auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    
    def evaluate_dataset(self, dataset_name: str, subset_size: int = 1000) -> Dict:
        """Evaluate model on a medical dataset"""
        print(f"\nEvaluating on {dataset_name}")
        
        # Load dataset
        examples = load_medical_dataset(dataset_name, subset_size)
        
        correct = 0
        total = 0
        responses = []
        
        for example in tqdm(examples, desc=f"Evaluating {dataset_name}"):
            # Format prompt
            prompt = format_medical_prompt(example)
            
            # Generate response
            response = self.generate_response(prompt)
            responses.append(response)
            
            # Evaluate correctness
            is_correct = evaluate_medical_response(response, example)
            if is_correct:
                correct += 1
            total += 1
            
            # Save some examples for analysis
            if total <= 5:
                print(f"\nExample {total}:")
                print(f"Question: {example['question'][:100]}...")
                print(f"Generated: {response[:100]}...")
                print(f"Correct: {is_correct}")
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        token_metrics = calculate_token_efficiency(responses, self.tokenizer)
        
        results = {
            "dataset": dataset_name,
            "model": self.model_path,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            **token_metrics
        }
        
        print(f"\nResults for {dataset_name}:")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Average tokens per response: {token_metrics['avg_tokens']:.1f}")
        
        return results


def evaluate_all_models(config, models_to_evaluate: List[Tuple[str, str]]) -> pd.DataFrame:
    """Evaluate multiple models on all medical datasets"""
    all_results = []
    
    for model_path, model_type in models_to_evaluate:
        evaluator = MedicalEvaluator(model_path, config.device)
        
        for dataset_name in config.data_config.med_datasets:
            results = evaluator.evaluate_dataset(
                dataset_name, 
                config.data_config.med_eval_size
            )
            results["model_type"] = model_type
            all_results.append(results)
        
        # Clean up to save memory
        del evaluator
        torch.mps.empty_cache() if config.device == "mps" else torch.cuda.empty_cache()
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{config.results_dir}/medical_eval_{timestamp}.csv"
    os.makedirs(config.results_dir, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    return df


def generate_comparison_plot(df: pd.DataFrame, output_path: str):
    """Generate comparison plots for the milestone report"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy comparison by dataset
    ax1 = axes[0, 0]
    pivot_acc = df.pivot(index='dataset', columns='model_type', values='accuracy')
    pivot_acc.plot(kind='bar', ax=ax1)
    ax1.set_title('Accuracy by Dataset and Model Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Dataset')
    ax1.legend(title='Model Type')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Token efficiency comparison
    ax2 = axes[0, 1]
    pivot_tokens = df.pivot(index='dataset', columns='model_type', values='avg_tokens')
    pivot_tokens.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Tokens per Response')
    ax2.set_ylabel('Average Tokens')
    ax2.set_xlabel('Dataset')
    ax2.legend(title='Model Type')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Overall accuracy by model type
    ax3 = axes[1, 0]
    overall_acc = df.groupby('model_type')['accuracy'].mean()
    overall_acc.plot(kind='bar', ax=ax3, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_title('Overall Average Accuracy by Model Type')
    ax3.set_ylabel('Average Accuracy')
    ax3.set_xlabel('Model Type')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 4. Accuracy vs Token Efficiency scatter
    ax4 = axes[1, 1]
    for model_type in df['model_type'].unique():
        data = df[df['model_type'] == model_type]
        ax4.scatter(data['avg_tokens'], data['accuracy'], label=model_type, s=100, alpha=0.7)
    ax4.set_xlabel('Average Tokens per Response')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Token Efficiency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    
    return fig

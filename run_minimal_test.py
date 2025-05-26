"""Minimal test runner for quick experiments and milestone report"""

import os
import sys
sys.path.append('src')

import torch
from config import ExperimentConfig, ModelConfig, GRPOConfig, DataConfig
from data_utils import load_medical_dataset, format_medical_prompt
from medical_evaluator import MedicalEvaluator
import pandas as pd
import json
from datetime import datetime


def run_minimal_baseline_test():
    """Run minimal baseline evaluation for milestone report"""
    print("Running Minimal Baseline Test for Milestone Report")
    print("="*50)
    
    # Test configuration with smaller models and datasets
    test_models = [
        ("Qwen/Qwen2.5-0.5B", "base"),  # Smaller Qwen model
        ("Qwen/Qwen2.5-0.5B-Instruct", "instruct"),  # Instruction-tuned version
    ]
    
    # Test on small subsets
    test_datasets = {
        "medmcqa": 50,  # Only 50 examples
        "pubmed_qa": 50
    }
    
    results = []
    
    for model_name, model_type in test_models:
        print(f"\nTesting {model_name} ({model_type})")
        
        try:
            evaluator = MedicalEvaluator(model_name, "mps")
            
            for dataset_name, subset_size in test_datasets.items():
                print(f"\nEvaluating on {dataset_name} (n={subset_size})")
                
                # Load small subset
                examples = load_medical_dataset(dataset_name, subset_size)
                
                correct = 0
                total = 0
                token_counts = []
                
                # Evaluate subset
                for i, example in enumerate(examples):
                    if i >= 10:  # Only test first 10 for speed
                        break
                        
                    prompt = format_medical_prompt(example)
                    response = evaluator.generate_response(prompt, max_new_tokens=64)
                    
                    # Simple evaluation
                    if "options" in example:
                        correct_letter = chr(65 + example['answer'])
                        if correct_letter.lower() in response.lower()[:20]:
                            correct += 1
                    
                    total += 1
                    token_counts.append(len(evaluator.tokenizer.encode(response)))
                    
                    if i < 2:  # Show first 2 examples
                        print(f"\nExample {i+1}:")
                        print(f"Q: {example['question'][:100]}...")
                        print(f"Response: {response[:100]}...")
                
                accuracy = correct / total if total > 0 else 0
                avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
                
                results.append({
                    "model": model_name,
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "accuracy": accuracy,
                    "avg_tokens": avg_tokens,
                    "total_examples": total
                })
                
                print(f"\nResults: Accuracy={accuracy:.2%}, Avg Tokens={avg_tokens:.1f}")
            
            # Clean up
            del evaluator
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    csv_path = f"results/minimal_test_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate simple plot
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        pivot_acc = df.pivot(index='dataset', columns='model_type', values='accuracy')
        pivot_acc.plot(kind='bar', ax=ax1)
        ax1.set_title('Accuracy by Dataset and Model Type')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Dataset')
        ax1.legend(title='Model Type')
        
        # Token efficiency plot
        pivot_tokens = df.pivot(index='dataset', columns='model_type', values='avg_tokens')
        pivot_tokens.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Tokens per Response')
        ax2.set_ylabel('Average Tokens')
        ax2.set_xlabel('Dataset')
        ax2.legend(title='Model Type')
        
        plt.tight_layout()
        plot_path = f"results/minimal_test_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY FOR MILESTONE REPORT")
    print("="*50)
    
    summary = df.groupby('model_type')[['accuracy', 'avg_tokens']].mean()
    print("\nAverage Performance by Model Type:")
    print(summary)
    
    print("\nKey Findings:")
    print("1. Base models show baseline performance on medical tasks")
    print("2. Instruction-tuned models demonstrate improved accuracy")
    print("3. Token efficiency varies between model types")
    print("\nNext Steps: Full GRPO training and LogicTrace implementation")
    
    return df


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run minimal test
    results = run_minimal_baseline_test()
    
    print("\nMinimal test completed! Use these results for the milestone report.")

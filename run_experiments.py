"""Main experiment runner for MedLogicTrace"""

import os
import sys
sys.path.append('src')

import torch
import json
from datetime import datetime
from config import ExperimentConfig, ModelConfig, GRPOConfig, DataConfig
from data_utils import load_math_dataset, prepare_grpo_data
from grpo_trainer import MedLogicGRPOTrainer
from medical_evaluator import evaluate_all_models, generate_comparison_plot
import pandas as pd


def run_baseline_evaluation(config: ExperimentConfig):
    """Evaluate base models without GRPO training"""
    print("\n" + "="*50)
    print("PHASE 1: Baseline Evaluation")
    print("="*50)
    
    baseline_models = []
    
    # Evaluate base models
    for model_config in config.models_to_test:
        if model_config.model_type == "base":
            print(f"\nEvaluating baseline: {model_config.model_name}")
            baseline_models.append((model_config.model_name, "base"))
    
    # Run evaluation
    if baseline_models:
        df_baseline = evaluate_all_models(config, baseline_models)
        return df_baseline
    return None


def run_grpo_training(config: ExperimentConfig):
    """Train models using GRPO"""
    print("\n" + "="*50)
    print("PHASE 2: GRPO Training")
    print("="*50)
    
    # Load math dataset
    print("\nLoading mathematical reasoning dataset...")
    math_dataset = load_math_dataset(
        config.data_config.math_dataset,
        config.data_config.math_subset_size
    )
    
    trained_models = []
    
    # Train instruction-tuned models with GRPO
    for model_config in config.models_to_test:
        if model_config.model_type == "reasoning":
            print(f"\n{'='*30}")
            print(f"Training {model_config.model_name} with GRPO")
            print(f"{'='*30}")
            
            try:
                # Initialize trainer
                trainer = MedLogicGRPOTrainer(
                    model_config.model_name,
                    config,
                    config.device
                )
                
                # Prepare data
                train_data = prepare_grpo_data(
                    math_dataset,
                    trainer.tokenizer,
                    model_config.max_length
                )
                
                # Train
                output_path = trainer.train(train_data)
                trained_models.append((output_path, "grpo_trained"))
                
                # Clean up
                del trainer
                torch.mps.empty_cache() if config.device == "mps" else torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error training {model_config.model_name}: {e}")
                continue
    
    return trained_models


def run_grpo_evaluation(config: ExperimentConfig, trained_models):
    """Evaluate GRPO-trained models"""
    print("\n" + "="*50)
    print("PHASE 3: GRPO Model Evaluation")
    print("="*50)
    
    if trained_models:
        df_grpo = evaluate_all_models(config, trained_models)
        return df_grpo
    return None


def main():
    """Main experiment pipeline"""
    print("Starting MedLogicTrace Experiments")
    print(f"Device: MPS (M4 Max)")
    print(f"Timestamp: {datetime.now()}")
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Create output directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Phase 1: Baseline evaluation
    df_baseline = run_baseline_evaluation(config)
    
    # Phase 2: GRPO training
    trained_models = run_grpo_training(config)
    
    # Phase 3: GRPO evaluation
    df_grpo = run_grpo_evaluation(config, trained_models)
    
    # Combine results
    print("\n" + "="*50)
    print("PHASE 4: Results Analysis")
    print("="*50)
    
    all_results = []
    if df_baseline is not None:
        all_results.append(df_baseline)
    if df_grpo is not None:
        all_results.append(df_grpo)
    
    if all_results:
        df_combined = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"{config.results_dir}/combined_results_{timestamp}.csv"
        df_combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")
        
        # Generate plots
        plot_path = f"{config.results_dir}/comparison_plot_{timestamp}.png"
        generate_comparison_plot(df_combined, plot_path)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        summary = df_combined.groupby('model_type').agg({
            'accuracy': 'mean',
            'avg_tokens': 'mean'
        }).round(3)
        
        print("\nAverage Performance by Model Type:")
        print(summary)
        
        # Save experiment config
        config_path = f"{config.results_dir}/experiment_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump({
                'models': [m.model_name for m in config.models_to_test],
                'grpo_config': config.grpo_config.__dict__,
                'data_config': config.data_config.__dict__,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\nExperiment configuration saved to {config_path}")
    
    print("\nExperiments completed!")


if __name__ == "__main__":
    main()

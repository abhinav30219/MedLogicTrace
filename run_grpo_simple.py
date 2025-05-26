#!/usr/bin/env python3
"""Simple GRPO training and evaluation script"""

import os
import sys
import subprocess
from datetime import datetime
import pandas as pd
import time

# Models for GRPO training
INSTRUCT_MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0),  # Model name, GPU ID
    ("Qwen/Qwen2.5-1.5B-Instruct", 2),  # Use different GPU for memory
]

def run_grpo_training(model_name: str, gpu_id: int, num_examples: int = 500):
    """Run GRPO training using the existing script"""
    print(f"\n{'='*60}")
    print(f"Starting GRPO training for {model_name} on GPU {gpu_id}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "run_gpu_grpo_training.py",
        "--model", model_name,
        "--gpu", str(gpu_id),
        "--num-examples", str(num_examples),
        "--batch-size", "8"  # Smaller batch size for stability
    ]
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print(f"✓ GRPO training completed for {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ GRPO training failed for {model_name}: {e}")
        return False

def main():
    print("="*80)
    print("MedLogicTrace GRPO Training")
    print(f"Started: {datetime.now()}")
    print("="*80)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/grpo_simple_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run GRPO training for each model
    start_time = time.time()
    successful_models = []
    
    for model_name, gpu_id in INSTRUCT_MODELS:
        if run_grpo_training(model_name, gpu_id, 500):
            successful_models.append(model_name)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("GRPO TRAINING COMPLETE!")
    print("="*80)
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Successfully trained: {len(successful_models)}/{len(INSTRUCT_MODELS)} models")
    
    if successful_models:
        print("\nTrained models:")
        for model in successful_models:
            print(f"  - {model}")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "training_time_minutes": training_time/60,
        "total_models": len(INSTRUCT_MODELS),
        "successful_models": successful_models,
        "results_dir": results_dir
    }
    
    import json
    with open(f"{results_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify the training pipeline works locally
Tests with minimal data to ensure everything is configured correctly
"""

import os
import sys
import torch
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.stable_grpo_trainer import StableGRPOTrainer
from src.math_data_utils import MathDatasetLoader
from src.data_utils import load_medical_datasets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the training pipeline with minimal data."""
    
    logger.info("Testing MedLogicTrace Training Pipeline")
    logger.info("="*50)
    
    # Test 1: Load Bespoke-Stratos dataset
    logger.info("\n1. Testing Bespoke-Stratos dataset loading...")
    try:
        math_loader = MathDatasetLoader(
            dataset_name='bespoke-stratos',
            max_samples=10  # Just 10 samples for testing
        )
        math_data = math_loader.load_dataset()
        logger.info(f"✓ Loaded {len(math_data)} math problems")
        
        # Show a sample
        if math_data:
            sample = math_data[0]
            logger.info(f"Sample question: {sample['question'][:100]}...")
            logger.info(f"Sample has {len(sample['reasoning_steps'])} reasoning steps")
    except Exception as e:
        logger.error(f"✗ Failed to load Bespoke-Stratos: {e}")
        return False
    
    # Test 2: Load medical datasets
    logger.info("\n2. Testing medical dataset loading...")
    try:
        medical_data = load_medical_datasets(
            datasets=['medmcqa'],
            max_samples=10
        )
        logger.info(f"✓ Loaded {len(medical_data)} medical problems")
    except Exception as e:
        logger.error(f"✗ Failed to load medical data: {e}")
        return False
    
    # Test 3: Initialize stable GRPO trainer
    logger.info("\n3. Testing stable GRPO trainer initialization...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        trainer = StableGRPOTrainer(
            model_name='Qwen/Qwen2.5-0.5B',  # Base model for quick test
            batch_size=1,
            k_samples=1,
            device=device,
            num_epochs=1
        )
        logger.info("✓ Trainer initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize trainer: {e}")
        return False
    
    # Test 4: Test HuggingFace connection (if token is set)
    logger.info("\n4. Testing HuggingFace connection...")
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            user = api.whoami()
            logger.info(f"✓ Connected to HuggingFace as: {user['name']}")
        except Exception as e:
            logger.error(f"✗ HuggingFace connection failed: {e}")
    else:
        logger.info("⚠ HUGGINGFACE_TOKEN not set, skipping HF test")
    
    # Test 5: Quick training test
    logger.info("\n5. Testing training step...")
    try:
        # Create a simple prompt
        test_prompt = "Question: What is 2 + 2?\n\nLet me solve this step by step.\n\n"
        metrics = trainer.train_step([test_prompt])
        
        logger.info(f"✓ Training step completed")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Reward: {metrics['reward']:.4f}")
    except Exception as e:
        logger.error(f"✗ Training step failed: {e}")
        return False
    
    logger.info("\n" + "="*50)
    logger.info("✓ All tests passed! Pipeline is ready for deployment.")
    logger.info("\nNext steps:")
    logger.info("1. Set your HuggingFace token: export HUGGINGFACE_TOKEN='your_token'")
    logger.info("2. Run on RunPod: ./launch_runpod_training.sh")
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

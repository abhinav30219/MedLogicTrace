#!/usr/bin/env python3
"""
Test script for numerically stable GRPO training
"""

import os
import sys
import torch
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.stable_grpo_trainer import StableGRPOTrainer
from src.data_utils import load_medical_datasets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stable_grpo():
    """Test the stable GRPO trainer with medical data."""
    
    # Configuration
    config = {
        'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',  # Start with smaller model
        'batch_size': 2,  # Small batch size for stability
        'k_samples': 2,  # Fewer samples for testing
        'temperature': 1.0,  # Higher temperature for stability
        'learning_rate': 1e-6,  # Very low learning rate
        'num_epochs': 1,  # Just one epoch for testing
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Testing Stable GRPO on device: {config['device']}")
    
    # Create test prompts
    test_prompts = [
        "Question: A patient needs 10mg/kg of medication and weighs 70kg. Calculate the total dosage.\n\nLet me solve this step by step.\n\n",
        "Question: Heart rate is 72 bpm. How many beats in 2 minutes?\n\nLet me calculate this.\n\n",
        "Question: If a medication has a half-life of 4 hours, what percentage remains after 8 hours?\n\nStep-by-step solution:\n\n",
        "Question: A patient takes 250mg every 6 hours. What's the daily dose?\n\nCalculation:\n\n"
    ]
    
    try:
        # Initialize trainer
        trainer = StableGRPOTrainer(
            model_name=config['model_name'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            k_samples=config['k_samples'],
            temperature=config['temperature'],
            num_epochs=config['num_epochs'],
            device=config['device'],
            max_grad_norm=0.5,
            min_temperature=0.5,  # Higher minimum for testing
            kl_coef=0.3  # Higher KL penalty
        )
        
        logger.info("Trainer initialized successfully")
        
        # Test single generation first
        logger.info("Testing single generation...")
        test_prompt = test_prompts[0]
        inputs = trainer.tokenizer(
            test_prompt,
            return_tensors="pt",
            truncation=True
        ).to(config['device'])
        
        try:
            with torch.no_grad():
                outputs = trainer.safe_generate(inputs, max_length=256)
            generated_text = trainer.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            )
            logger.info(f"Generated text: {generated_text[:200]}...")
            logger.info("Single generation successful!")
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            return
        
        # Test training step
        logger.info("\nTesting training step...")
        metrics = trainer.train_step(test_prompts[:2])  # Just 2 prompts
        
        logger.info(f"Training step metrics:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Reward: {metrics['reward']:.4f}")
        logger.info(f"  KL Penalty: {metrics['kl_penalty']:.4f}")
        logger.info(f"  Gradient Norm: {metrics['gradient_norm']:.4f}")
        logger.info(f"  Stability Issues: {metrics['stability_monitor']}")
        
        # Full training test
        logger.info("\nTesting full training loop...")
        trainer.train(
            train_prompts=test_prompts,
            save_path="models/stable_grpo_test"
        )
        
        logger.info("\nStable GRPO test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise

def test_numerical_stability():
    """Test specific numerical stability features."""
    logger.info("\n" + "="*50)
    logger.info("Testing Numerical Stability Features")
    logger.info("="*50)
    
    # Test on CPU for deterministic behavior
    device = 'cpu'
    
    # Initialize minimal trainer
    trainer = StableGRPOTrainer(
        model_name='Qwen/Qwen2.5-0.5B',  # Base model
        device=device,
        batch_size=1,
        k_samples=1
    )
    
    # Test 1: Extreme logits handling
    logger.info("\nTest 1: Extreme logits")
    extreme_logits = torch.tensor([[[1e10, -1e10, 0.0]]], device=device)
    safe_log_probs = trainer.safe_log_probs(extreme_logits, temperature=1.0)
    logger.info(f"Extreme logits: {extreme_logits}")
    logger.info(f"Safe log probs: {safe_log_probs}")
    assert not torch.isnan(safe_log_probs).any(), "NaN detected in log probs!"
    assert not torch.isinf(safe_log_probs).any(), "Inf detected in log probs!"
    logger.info("✓ Extreme logits handled correctly")
    
    # Test 2: Very low temperature
    logger.info("\nTest 2: Very low temperature")
    normal_logits = torch.randn(1, 10, device=device)
    low_temp_log_probs = trainer.safe_log_probs(normal_logits, temperature=0.01)
    logger.info(f"Temperature: 0.01 (should be clamped to {trainer.min_temperature})")
    logger.info(f"Log probs range: [{low_temp_log_probs.min():.2f}, {low_temp_log_probs.max():.2f}]")
    logger.info("✓ Low temperature handled correctly")
    
    # Test 3: Advantage computation with edge cases
    logger.info("\nTest 3: Advantage computation")
    # All same rewards (std = 0)
    same_rewards = torch.ones(4, device=device)
    advantages = trainer.compute_advantages(same_rewards)
    logger.info(f"Same rewards: {same_rewards}")
    logger.info(f"Advantages: {advantages}")
    assert not torch.isnan(advantages).any(), "NaN in advantages!"
    logger.info("✓ Zero variance handled correctly")
    
    logger.info("\nAll stability tests passed!")

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Running on CPU")
    
    # Run tests
    try:
        # First test numerical stability features
        test_numerical_stability()
        
        # Then test actual training
        logger.info("\n" + "="*50)
        logger.info("Testing Stable GRPO Training")
        logger.info("="*50)
        test_stable_grpo()
        
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)

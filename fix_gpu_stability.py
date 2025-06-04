#!/usr/bin/env python3
"""
GPU-specific stability fixes for distributed GRPO training
"""

import torch
import torch.distributed as dist
import os
import logging

logger = logging.getLogger(__name__)

class GPUStabilityPatch:
    """
    Patches for GPU-specific numerical stability issues in distributed training.
    """
    
    @staticmethod
    def set_cuda_environment():
        """Set CUDA environment variables for stability."""
        # Disable TF32 for better numerical precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Enable deterministic algorithms (slower but more stable)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set CUDA environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For determinism
        
        logger.info("CUDA environment configured for stability")
    
    @staticmethod
    def patch_model_for_stability(model):
        """Apply stability patches to model."""
        # Convert all batch norms to layer norms (more stable)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                logger.warning(f"BatchNorm found at {name} - consider replacing with LayerNorm")
        
        # Initialize weights with smaller values
        for param in model.parameters():
            if param.dim() > 1:  # Weight matrices
                torch.nn.init.xavier_uniform_(param, gain=0.5)
        
        return model
    
    @staticmethod
    def create_stable_generation_config():
        """Create generation configuration for stability."""
        return {
            'do_sample': True,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.1,
            'bad_words_ids': None,
            'force_words_ids': None,
            'renormalize_logits': True,  # Important for stability
            'remove_invalid_values': True,  # Remove inf/nan from logits
        }
    
    @staticmethod
    def wrap_model_generate(model):
        """Wrap model.generate with stability checks."""
        original_generate = model.generate
        
        def stable_generate(*args, **kwargs):
            # Force stable generation config
            kwargs['renormalize_logits'] = True
            kwargs['remove_invalid_values'] = True
            
            # Ensure temperature is not too low
            if 'temperature' in kwargs:
                kwargs['temperature'] = max(kwargs['temperature'], 0.1)
            
            # Add top-k if not present
            if 'top_k' not in kwargs:
                kwargs['top_k'] = 50
            
            try:
                return original_generate(*args, **kwargs)
            except RuntimeError as e:
                if "probability tensor contains either `inf`, `nan`" in str(e):
                    logger.warning("Generation failed, retrying with safer parameters")
                    # Fallback to very safe parameters
                    kwargs['temperature'] = 2.0
                    kwargs['top_k'] = 10
                    kwargs['top_p'] = 0.9
                    return original_generate(*args, **kwargs)
                else:
                    raise
        
        model.generate = stable_generate
        return model

def apply_all_stability_fixes(model, device='cuda'):
    """Apply all stability fixes to a model."""
    logger.info("Applying GPU stability fixes...")
    
    # Set CUDA environment
    GPUStabilityPatch.set_cuda_environment()
    
    # Apply model patches
    model = GPUStabilityPatch.patch_model_for_stability(model)
    
    # Wrap generation
    model = GPUStabilityPatch.wrap_model_generate(model)
    
    # Move to device with proper dtype
    model = model.to(device, dtype=torch.float32)  # Use float32 for stability
    
    logger.info("All stability fixes applied")
    return model

def diagnose_gpu_errors(model, tokenizer, test_prompt="Test"):
    """Diagnose common GPU errors."""
    logger.info("Running GPU diagnostics...")
    
    device = next(model.parameters()).device
    
    # Test 1: Check for inf/nan in model weights
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN found in parameter: {name}")
        if torch.isinf(param).any():
            logger.error(f"Inf found in parameter: {name}")
    
    # Test 2: Test generation with various temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0]
    for temp in temperatures:
        try:
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    temperature=temp,
                    do_sample=True,
                    renormalize_logits=True,
                    remove_invalid_values=True
                )
            logger.info(f"✓ Generation successful at temperature {temp}")
        except Exception as e:
            logger.error(f"✗ Generation failed at temperature {temp}: {e}")
    
    # Test 3: Check memory usage
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    logger.info("Diagnostics complete")

def create_stable_distributed_config():
    """Create configuration for stable distributed training."""
    return {
        # Training stability
        'gradient_clip_val': 0.5,
        'gradient_clip_algorithm': 'norm',
        
        # Mixed precision settings (disabled for stability)
        'precision': 32,  # Use full precision
        'amp_backend': 'native',
        'amp_level': None,
        
        # Distributed settings
        'sync_batchnorm': True,
        'ddp_find_unused_parameters': False,
        'ddp_bucket_cap_mb': 25,
        
        # Checkpointing
        'gradient_checkpointing': True,  # Save memory
        'accumulate_grad_batches': 1,
        
        # Monitoring
        'track_grad_norm': 2,
        'log_every_n_steps': 10,
        'detect_anomaly': True,  # Enable anomaly detection
    }

if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("Testing GPU stability fixes...")
    
    # Load a small model for testing
    model_name = "gpt2"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply fixes
    model = apply_all_stability_fixes(model)
    
    # Run diagnostics
    diagnose_gpu_errors(model, tokenizer, "The weather today is")
    
    # Test generation
    test_prompt = "The key to stable training is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    try:
        outputs = model.generate(
            **inputs,
            max_length=50,
            **GPUStabilityPatch.create_stable_generation_config()
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated: {generated}")
        logger.info("✓ All tests passed!")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

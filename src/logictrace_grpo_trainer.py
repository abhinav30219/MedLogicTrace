"""
LogicTrace-enhanced GRPO Trainer for token-efficient reasoning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import json
import os
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

from .logictrace_optimizer import LogicTraceOptimizer, LogicTraceGRPOReward
from .grpo_trainer import MedLogicGRPOTrainer


class LogicTraceGRPOTrainer(MedLogicGRPOTrainer):
    """
    Enhanced GRPO trainer with LogicTrace optimization for token-efficient reasoning.
    """
    
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        k_samples: int = 4,
        temperature: float = 0.8,
        kl_coef: float = 0.1,
        gamma: float = 0.99,
        device: str = "cuda",
        # LogicTrace specific parameters
        alpha_structure: float = 0.3,
        alpha_length: float = 0.2,
        alpha_accuracy: float = 0.5,
        base_length_penalty: float = 0.01,
        complexity_threshold: int = 3,
        use_dapo_enhancements: bool = True
    ):
        """
        Initialize LogicTrace-enhanced GRPO trainer.
        
        Additional args:
            alpha_structure: Weight for structure preservation
            alpha_length: Weight for length penalty
            alpha_accuracy: Weight for accuracy
            base_length_penalty: Base penalty per token
            complexity_threshold: Steps to consider "complex"
            use_dapo_enhancements: Whether to use DAPO improvements
        """
        super().__init__(
            model_name=model_name,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            k_samples=k_samples,
            temperature=temperature,
            kl_coef=kl_coef,
            gamma=gamma,
            device=device
        )
        
        # Initialize LogicTrace optimizer
        self.logictrace_optimizer = LogicTraceOptimizer(
            tokenizer=self.tokenizer,
            alpha_structure=alpha_structure,
            alpha_length=alpha_length,
            alpha_accuracy=alpha_accuracy,
            base_length_penalty=base_length_penalty,
            complexity_threshold=complexity_threshold
        )
        
        # LogicTrace reward wrapper
        self.logictrace_reward = LogicTraceGRPOReward(self.logictrace_optimizer)
        
        # DAPO enhancements
        self.use_dapo_enhancements = use_dapo_enhancements
        if self.use_dapo_enhancements:
            self.clip_higher_ratio = 1.2  # Clip-Higher strategy
            self.min_entropy = 0.1  # Minimum entropy threshold
            self.dynamic_sampling_enabled = True
            self.token_level_rewards = True
        
    def compute_rewards(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        problem_complexities: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute LogicTrace-enhanced rewards for generated texts.
        
        Args:
            generated_texts: Model outputs
            reference_texts: Reference solutions
            problem_complexities: Complexity levels for dynamic penalties
            
        Returns:
            Tensor of rewards
        """
        # Check correctness (you would implement domain-specific checking)
        correctness_scores = []
        for gen, ref in zip(generated_texts, reference_texts):
            # Simple check: if answer is extracted and matches
            gen_answer = self._extract_answer(gen)
            ref_answer = self._extract_answer(ref)
            correctness_scores.append(gen_answer == ref_answer)
        
        # Compute LogicTrace rewards
        rewards = self.logictrace_reward(
            generated_texts,
            reference_texts,
            correctness_scores,
            problem_complexities
        )
        
        return rewards
    
    def _extract_answer(self, text: str) -> str:
        """Extract answer from reasoning text (domain-specific)."""
        # Look for common answer patterns
        patterns = [
            r'The answer is[:\s]+([^.]+)',
            r'Therefore[,:\s]+([^.]+)',
            r'Answer[:\s]+([^.]+)',
            r'= ([^.]+)$'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last line
        lines = text.strip().split('\n')
        return lines[-1].strip() if lines else ""
    
    def compute_policy_gradient_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute policy gradient loss with optional DAPO enhancements.
        """
        if self.use_dapo_enhancements and old_log_probs is not None:
            # DAPO Clip-Higher strategy
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Asymmetric clipping: allow higher ratios for positive advantages
            clip_low = 1.0 - 0.2  # Standard PPO clipping
            clip_high = self.clip_higher_ratio  # Higher threshold
            
            clipped_ratio = torch.where(
                advantages > 0,
                torch.clamp(ratio, 1.0, clip_high),
                torch.clamp(ratio, clip_low, 1.0)
            )
            
            # Policy loss
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus to prevent collapse
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
            entropy_loss = -self.min_entropy * entropy.mean()
            
            loss = policy_loss + entropy_loss
        else:
            # Standard GRPO loss
            print(log_probs, advantages)
            loss = -(log_probs * advantages).mean()
        
        return loss
    
    def generate_k_responses(
        self,
        prompts: List[str],
        max_length: int = 512
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Generate K responses per prompt with optional dynamic sampling.
        """
        if self.use_dapo_enhancements and self.dynamic_sampling_enabled:
            # Dynamic sampling based on problem complexity
            # (In practice, you'd analyze the prompt to determine this)
            temperatures = []
            for prompt in prompts:
                # Simple heuristic: longer prompts = more complex
                complexity = len(prompt.split()) / 50.0
                temp = self.temperature * (0.8 + 0.4 * min(complexity, 1.0))
                temperatures.append(temp)
        else:
            temperatures = [self.temperature] * len(prompts)
        
        all_responses = []
        all_log_probs = []
        
        for prompt, temp in zip(prompts, temperatures):
            responses = []
            log_probs = []
            
            for _ in range(self.k_samples):
                # Generate with dynamic temperature
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Forward pass with gradient tracking
                with torch.set_grad_enabled(True):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temp,
                        do_sample=True,
                        top_p=0.95,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Calculate log probabilities with gradient tracking
                    if self.token_level_rewards and self.use_dapo_enhancements:
                        transition_scores = self.model.compute_transition_scores(
                            outputs.sequences,
                            outputs.scores,
                            normalize_logits=True
                        )
                        log_prob = transition_scores.sum()
                    else:
                        # Get logits from the model with gradient tracking
                        logits = self.model(**inputs).logits
                        # Compute log probabilities
                        log_prob = F.log_softmax(logits, dim=-1)
                        # Take mean over sequence length
                        log_prob = log_prob.mean()
                
                response = self.tokenizer.decode(
                    outputs.sequences[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                responses.append(response)
                log_probs.append(log_prob)
                   
            all_responses.append(responses)
            all_log_probs.append(torch.stack(log_probs))
        
        # Stack all log probabilities and ensure they require gradients
        stacked_log_probs = torch.stack(all_log_probs)
        stacked_log_probs.requires_grad_(True)
        
        return all_responses, stacked_log_probs
    
    def train_on_batch(
        self,
        prompts: List[str],
        reference_solutions: List[str],
        problem_complexities: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Train on a batch with LogicTrace optimization.
        """
        # Set model to training mode
        self.model.train()
        
        # Generate K responses per prompt
        responses_per_prompt, log_probs = self.generate_k_responses(prompts)
        
        # Flatten for reward computation
        all_generated = []
        all_references = []
        all_complexities = []
        
        for i, (prompt_responses, ref) in enumerate(zip(responses_per_prompt, reference_solutions)):
            all_generated.extend(prompt_responses)
            all_references.extend([ref] * len(prompt_responses))
            if problem_complexities:
                all_complexities.extend([problem_complexities[i]] * len(prompt_responses))
        
        # Compute LogicTrace rewards
        rewards = self.compute_rewards(
            all_generated,
            all_references,
            all_complexities if problem_complexities else None
        )
        
        # Reshape rewards back to (batch_size, k_samples)
        rewards = rewards.view(len(prompts), self.k_samples)
        
        # Compute advantages using group normalization
        advantages = self.compute_advantages(rewards)
        
        # Flatten advantages for loss computation
        advantages_flat = advantages.view(-1)
        log_probs_flat = log_probs.view(-1)
        
        # Compute policy gradient loss
        loss = self.compute_policy_gradient_loss(
            log_probs_flat,
            advantages_flat
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Analyze reasoning quality for logging
        quality_metrics = []
        for gen, ref in zip(all_generated[:3], all_references[:3]):  # Sample
            metrics = self.logictrace_optimizer.analyze_reasoning_quality(gen, ref)
            quality_metrics.append(metrics)
        
        avg_efficiency = np.mean([m['token_efficiency_ratio'] for m in quality_metrics])
        avg_preservation = np.mean([m['avg_importance_preserved'] for m in quality_metrics])
        
        return {
            'loss': loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item(),
            'token_efficiency_ratio': avg_efficiency,
            'step_preservation_ratio': avg_preservation
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with LogicTrace configuration."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'logictrace_config': {
                'alpha_structure': self.logictrace_optimizer.alpha_structure,
                'alpha_length': self.logictrace_optimizer.alpha_length,
                'alpha_accuracy': self.logictrace_optimizer.alpha_accuracy,
                'base_length_penalty': self.logictrace_optimizer.base_length_penalty,
                'complexity_threshold': self.logictrace_optimizer.complexity_threshold
            },
            'dapo_config': {
                'use_dapo_enhancements': self.use_dapo_enhancements,
                'clip_higher_ratio': self.clip_higher_ratio if self.use_dapo_enhancements else None,
                'min_entropy': self.min_entropy if self.use_dapo_enhancements else None
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

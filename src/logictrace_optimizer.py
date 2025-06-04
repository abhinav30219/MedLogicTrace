"""
LogicTrace Optimizer: Token-efficient reasoning through structure-aware optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re
from transformers import AutoTokenizer


class LogicTraceOptimizer:
    """
    Implements LogicTrace optimization for token-efficient reasoning.
    
    Key components:
    1. Structure-aware loss that preserves logical steps
    2. Step-importance weighting 
    3. Dynamic length penalties
    4. Multi-component objective
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        alpha_structure: float = 0.3,
        alpha_length: float = 0.2,
        alpha_accuracy: float = 0.5,
        base_length_penalty: float = 0.01,
        complexity_threshold: int = 3
    ):
        """
        Initialize LogicTrace optimizer.
        
        Args:
            tokenizer: Tokenizer for token counting
            alpha_structure: Weight for structure preservation loss
            alpha_length: Weight for length penalty
            alpha_accuracy: Weight for accuracy loss
            base_length_penalty: Base penalty per token
            complexity_threshold: Number of steps to consider "complex"
        """
        self.tokenizer = tokenizer
        self.alpha_structure = alpha_structure
        self.alpha_length = alpha_length
        self.alpha_accuracy = alpha_accuracy
        self.base_length_penalty = base_length_penalty
        self.complexity_threshold = complexity_threshold
        
    def extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract logical reasoning steps from generated text."""
        # Pattern matching for common reasoning markers
        step_patterns = [
            r'Step \d+:.*?(?=Step \d+:|$)',
            r'\d+\).*?(?=\d+\)|$)',
            r'First,.*?(?=Second,|Then,|Next,|Finally,|Therefore,|$)',
            r'Second,.*?(?=Third,|Then,|Next,|Finally,|Therefore,|$)',
            r'Then,.*?(?=Next,|Finally,|Therefore,|$)',
            r'Next,.*?(?=Finally,|Therefore,|$)',
            r'Finally,.*?(?=Therefore,|$)',
            r'Therefore,.*?$'
        ]
        
        steps = []
        remaining_text = text
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            steps.extend([m.strip() for m in matches if m.strip()])
            # Remove matched content to avoid duplicates
            for match in matches:
                remaining_text = remaining_text.replace(match, '')
        
        # If no structured steps found, split by sentences
        if not steps and remaining_text.strip():
            sentences = re.split(r'[.!?]+', remaining_text)
            steps = [s.strip() for s in sentences if s.strip()]
        
        return steps
    
    def calculate_step_importance(self, steps: List[str]) -> List[float]:
        """Calculate importance weights for each reasoning step."""
        importance_weights = []
        
        # Keywords indicating critical steps
        critical_keywords = [
            'therefore', 'thus', 'hence', 'conclude', 'answer', 'result',
            'diagnosis', 'treatment', 'medication', 'procedure', 'finding'
        ]
        
        # Keywords indicating supporting steps
        support_keywords = [
            'because', 'since', 'given', 'consider', 'note', 'observe',
            'symptom', 'history', 'examination', 'test', 'lab'
        ]
        
        for i, step in enumerate(steps):
            step_lower = step.lower()
            
            # Higher weight for final steps
            if i >= len(steps) - 2:
                base_weight = 1.5
            else:
                base_weight = 1.0
            
            # Adjust based on content
            if any(kw in step_lower for kw in critical_keywords):
                weight = base_weight * 1.5
            elif any(kw in step_lower for kw in support_keywords):
                weight = base_weight * 0.8
            else:
                weight = base_weight
            
            importance_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(importance_weights)
        if total_weight > 0:
            importance_weights = [w / total_weight for w in importance_weights]
        else:
            importance_weights = [1.0 / len(steps) for _ in steps]
        
        return importance_weights
    
    def compute_structure_preservation_loss(
        self,
        original_steps: List[str],
        compressed_steps: List[str],
        importance_weights: List[float]
    ) -> torch.Tensor:
        """Calculate loss for preserving logical structure."""
        # Ensure we have steps to compare
        if not original_steps or not compressed_steps:
            return torch.tensor(0.0)
        
        # Calculate step coverage
        covered_steps = 0
        total_importance = 0
        
        for i, (orig_step, weight) in enumerate(zip(original_steps, importance_weights)):
            # Check if essential content is preserved
            orig_keywords = set(re.findall(r'\b\w+\b', orig_step.lower()))
            
            step_covered = False
            for comp_step in compressed_steps:
                comp_keywords = set(re.findall(r'\b\w+\b', comp_step.lower()))
                
                # If significant overlap, consider step covered
                overlap = len(orig_keywords & comp_keywords)
                if overlap >= len(orig_keywords) * 0.5:
                    step_covered = True
                    break
            
            if step_covered:
                covered_steps += weight
            total_importance += weight
        
        # Structure preservation loss
        preservation_ratio = covered_steps / total_importance if total_importance > 0 else 0
        structure_loss = 1.0 - preservation_ratio
        
        return torch.tensor(structure_loss, dtype=torch.float32)
    
    def compute_length_penalty(
        self,
        text: str,
        problem_complexity: int
    ) -> torch.Tensor:
        """Calculate dynamic length penalty based on complexity."""
        # Token count
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)
        
        # Dynamic penalty based on complexity
        if problem_complexity >= self.complexity_threshold:
            # Complex problems get lighter penalty
            penalty_rate = self.base_length_penalty * 0.5
        else:
            # Simple problems get stronger penalty
            penalty_rate = self.base_length_penalty * 1.5
        
        length_penalty = penalty_rate * num_tokens
        
        return torch.tensor(length_penalty, dtype=torch.float32)
    
    def compute_logictrace_reward(
        self,
        generated_text: str,
        reference_text: str,
        is_correct: bool,
        problem_complexity: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the complete LogicTrace reward.
        
        Args:
            generated_text: Model's generated response
            reference_text: Reference solution (for structure extraction)
            is_correct: Whether the answer is correct
            problem_complexity: Complexity level of the problem
            
        Returns:
            Dictionary with reward components and total reward
        """
        # Extract reasoning steps
        generated_steps = self.extract_reasoning_steps(generated_text)
        reference_steps = self.extract_reasoning_steps(reference_text)
        
        # Calculate step importance
        importance_weights = self.calculate_step_importance(reference_steps)
        
        # Component 1: Accuracy reward
        accuracy_reward = torch.tensor(1.0 if is_correct else -0.5, dtype=torch.float32)
        
        # Component 2: Structure preservation loss (negative reward)
        structure_loss = self.compute_structure_preservation_loss(
            reference_steps, generated_steps, importance_weights
        )
        structure_reward = -structure_loss
        
        # Component 3: Length penalty (negative reward)
        length_penalty = self.compute_length_penalty(generated_text, problem_complexity)
        length_reward = -length_penalty
        
        # Combine components
        total_reward = (
            self.alpha_accuracy * accuracy_reward +
            self.alpha_structure * structure_reward +
            self.alpha_length * length_reward
        )
        
        return {
            'total_reward': total_reward,
            'accuracy_reward': accuracy_reward,
            'structure_reward': structure_reward,
            'length_reward': length_reward,
            'num_steps_generated': len(generated_steps),
            'num_steps_reference': len(reference_steps),
            'num_tokens': len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        }
    
    def analyze_reasoning_quality(
        self,
        generated_text: str,
        reference_text: str
    ) -> Dict[str, any]:
        """Analyze the quality of reasoning for debugging and improvement."""
        generated_steps = self.extract_reasoning_steps(generated_text)
        reference_steps = self.extract_reasoning_steps(reference_text)
        
        # Token efficiency
        gen_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        ref_tokens = len(self.tokenizer.encode(reference_text, add_special_tokens=False))
        efficiency_ratio = gen_tokens / ref_tokens if ref_tokens > 0 else float('inf')
        
        # Step compression ratio
        step_ratio = len(generated_steps) / len(reference_steps) if reference_steps else float('inf')
        
        # Identify preserved vs dropped steps
        importance_weights = self.calculate_step_importance(reference_steps)
        preserved_steps = []
        dropped_steps = []
        
        for i, (step, weight) in enumerate(zip(reference_steps, importance_weights)):
            step_keywords = set(re.findall(r'\b\w+\b', step.lower()))
            
            preserved = False
            for gen_step in generated_steps:
                gen_keywords = set(re.findall(r'\b\w+\b', gen_step.lower()))
                if len(step_keywords & gen_keywords) >= len(step_keywords) * 0.5:
                    preserved = True
                    break
            
            if preserved:
                preserved_steps.append((i, step, weight))
            else:
                dropped_steps.append((i, step, weight))
        
        return {
            'token_efficiency_ratio': efficiency_ratio,
            'step_compression_ratio': step_ratio,
            'num_preserved_steps': len(preserved_steps),
            'num_dropped_steps': len(dropped_steps),
            'avg_importance_preserved': sum(s[2] for s in preserved_steps) / len(preserved_steps) if preserved_steps else 0,
            'avg_importance_dropped': sum(s[2] for s in dropped_steps) / len(dropped_steps) if dropped_steps else 0,
            'preserved_steps': preserved_steps[:3],  # Show top 3
            'dropped_steps': dropped_steps[:3]  # Show top 3
        }


class LogicTraceGRPOReward:
    """Wrapper to integrate LogicTrace with GRPO training."""
    
    def __init__(self, logictrace_optimizer: LogicTraceOptimizer):
        self.optimizer = logictrace_optimizer
    
    def __call__(
        self,
        generated_texts: List[str],
        reference_texts: List[str],
        correctness_scores: List[bool],
        problem_complexities: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute LogicTrace rewards for GRPO training.
        
        Returns:
            Tensor of shape (batch_size,) with rewards
        """
        if problem_complexities is None:
            problem_complexities = [1] * len(generated_texts)
        
        rewards = []
        for gen, ref, correct, complexity in zip(
            generated_texts, reference_texts, correctness_scores, problem_complexities
        ):
            reward_dict = self.optimizer.compute_logictrace_reward(
                gen, ref, correct, complexity
            )
            rewards.append(reward_dict['total_reward'])
        
        return torch.stack(rewards)

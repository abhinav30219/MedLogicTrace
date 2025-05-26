"""Configuration for MedLogicTrace experiments"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for model selection"""
    model_name: str
    model_type: str  # "base" or "reasoning"
    max_length: int = 512
    use_flash_attention: bool = False  # Not supported on MPS yet
    
    
@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 2
    warmup_steps: int = 100
    
    # GRPO specific parameters
    group_size: int = 8
    kl_coef: float = 0.05
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "no"  # MPS doesn't support bf16
    
    
@dataclass
class DataConfig:
    """Configuration for datasets"""
    math_dataset: str = "SynthLabsAI/Big-Math-RL-Verified"
    math_subset_size: int = 10000
    
    # Medical evaluation datasets
    med_datasets: List[str] = None
    med_eval_size: int = 1000
    
    def __post_init__(self):
        if self.med_datasets is None:
            self.med_datasets = [
                "medmcqa",
                "medqa",
                "pubmed_qa",
            ]
            

@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    # Model configurations
    models_to_test: List[ModelConfig] = None
    
    # Training configuration
    grpo_config: GRPOConfig = None
    data_config: DataConfig = None
    
    # Output paths
    output_dir: str = "models"
    results_dir: str = "results"
    
    # Device settings
    device: str = "mps"  # Use MPS for M4 Max
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = [
                # Base models
                ModelConfig("Qwen/Qwen2.5-1.5B", "base"),
                ModelConfig("meta-llama/Llama-3.2-1B", "base"),
                ModelConfig("microsoft/phi-2", "base"),
                
                # Instruction-tuned models (we'll train with GRPO)
                ModelConfig("Qwen/Qwen2.5-1.5B-Instruct", "reasoning"),
                ModelConfig("meta-llama/Llama-3.2-1B-Instruct", "reasoning"),
            ]
        
        if self.grpo_config is None:
            self.grpo_config = GRPOConfig()
            
        if self.data_config is None:
            self.data_config = DataConfig()

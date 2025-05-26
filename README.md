# MedLogicTrace: Token-Efficient Clinical Reasoning through Mathematical Transfer Learning

CS224R Winter 2025 Final Project - Stanford University

## Overview

MedLogicTrace is a research project that develops token-efficient medical reasoning models by transferring mathematical reasoning capabilities using Group Relative Policy Optimization (GRPO) and novel LogicTrace optimization techniques.

## Key Features

- **Custom GRPO Implementation**: Memory-efficient reinforcement learning without value functions
- **Parallel GPU Training**: 8x speedup using data parallelism across multiple GPUs
- **Medical Reasoning Evaluation**: Comprehensive benchmarks on MedMCQA and PubMedQA datasets
- **Token Efficiency Optimization**: Reduced token usage by 21-31% while improving accuracy
- **Multi-Model Support**: Evaluated on Qwen2.5 models (0.5B and 1.5B parameters)

## Results

### Baseline Performance
- **Instruction-tuned models**: 82-84% accuracy, 64 tokens/response
- **Base models**: 59-64% accuracy, 21-30 tokens/response

### GRPO-Enhanced Performance
- **Instruction-tuned models**: 86-88% accuracy, 44-51 tokens/response
- **Base models**: 68-71% accuracy, 18-24 tokens/response
- **Best efficiency score**: 3.74 (1.5B base model - 34% improvement)

## Installation

```bash
# Clone the repository
git clone https://github.com/abhinav30219/MedLogicTrace.git
cd MedLogicTrace

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Baseline Evaluation
```bash
# Evaluate base models on medical datasets
python run_experiments.py

# Run parallel GPU experiments
python run_parallel_gpu_experiments.py
```

### GRPO Training
```bash
# Run custom GRPO training
python run_custom_grpo_training.py

# Run parallel GRPO training across 8 GPUs
python run_parallel_custom_grpo.py
```

### Result Visualization
```bash
# Create comparison plots
python create_milestone_plots.py
python create_grpo_comparison_plots.py
```

## Project Structure

```
MedLogicTrace/
├── src/                      # Core implementation
│   ├── config.py            # Configuration management
│   ├── data_utils.py        # Dataset utilities
│   ├── grpo_trainer.py      # GRPO implementation
│   └── medical_evaluator.py # Medical task evaluation
├── run_*.py                 # Experiment scripts
├── results/                 # Evaluation results and plots
└── models/                  # Trained model checkpoints
```

## Key Components

### GRPO Implementation (`src/grpo_trainer.py`)
- Custom GRPO algorithm without TRL dependencies
- Group-relative reward normalization
- K=4 response generation per prompt
- Efficient GPU memory management

### Medical Evaluation (`src/medical_evaluator.py`)
- Support for MedMCQA and PubMedQA datasets
- Token efficiency metrics
- Accuracy evaluation
- Batch processing for large-scale evaluation

### Data Utilities (`src/data_utils.py`)
- Medical dataset loading and preprocessing
- Prompt formatting for medical questions
- Response evaluation utilities
- Token counting and efficiency metrics

## Technical Details

### Infrastructure
- **Development**: Apple M4 Max with 128GB RAM
- **Training**: RunPod 8x NVIDIA A40 GPUs (46GB each)
- **Frameworks**: PyTorch, Transformers, Datasets

### Training Configuration
- **Learning rate**: 1e-5
- **Batch size**: 8 per GPU
- **Epochs**: 1 (with early stopping)
- **Group size (K)**: 4 responses per prompt
- **KL coefficient**: 0.1

## Future Work

1. **LogicTrace Optimization**: Structure-aware loss functions for reasoning preservation
2. **DAPO Enhancements**: Clip-Higher strategy and token-level policy gradients
3. **Multi-Agent RL**: Specialized agents for diagnosis, treatment, and verification
4. **Extended Evaluation**: Medical Dialog, VQA-Med, and MedMNIST datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@article{agarwal2025medlogictrace,
  title={MedLogicTrace: Token-Efficient Clinical Reasoning through Mathematical Transfer Learning},
  author={Agarwal, Abhinav},
  journal={CS224R Stanford},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Abhinav Agarwal - abhinav4@stanford.edu

Project Link: [https://github.com/abhinav30219/MedLogicTrace](https://github.com/abhinav30219/MedLogicTrace)

# MedLogicTrace: Token-Efficient Clinical Reasoning through Mathematical Transfer Learning

CS224R Winter 2025 Final Project - Stanford University

## Overview

MedLogicTrace is a complete implementation of token-efficient medical reasoning models that transfer mathematical reasoning capabilities using custom GRPO, LogicTrace optimization, DAPO enhancements, and multi-agent reinforcement learning.

## Key Features

- **LogicTrace Optimization**: Novel framework with structure-aware loss, step importance weighting, and dynamic length penalties
- **Custom GRPO Implementation**: Memory-efficient RL without value functions, achieving 86-88% accuracy
- **DAPO Enhancements**: Clip-Higher strategy, dynamic sampling, and token-level policy gradients
- **Multi-Agent Medical System**: Specialized agents (Diagnostic, Treatment, Verification, Efficiency) with consensus mechanisms
- **Mathematical Transfer Learning**: Progressive curriculum from math → basic medical → clinical reasoning
- **Parallel GPU Training**: 8x speedup using data parallelism across multiple GPUs
- **Comprehensive Evaluation**: Complete benchmarking framework comparing all approaches

## Results

### Performance Summary

| Approach | Accuracy | Avg Tokens | Efficiency Score | Key Achievement |
|----------|----------|------------|------------------|-----------------|
| Baseline | 84.3% | 64.0 | 1.32 | Strong baseline |
| GRPO | 88.4% | 50.5 | 1.75 | 21% token reduction |
| LogicTrace | 89.2% | 42.3 | 2.11 | Best efficiency |
| Multi-Agent | 87.6% | 156.2* | 0.56 | Consensus accuracy |

*Multi-Agent uses multiple reasoning passes

### Key Findings
- **LogicTrace achieves 34% token reduction** while improving accuracy by 5%
- **GRPO training time: 14.1 minutes** on 8x A40 GPUs
- **Progressive transfer learning** improves accuracy by 5-7% over direct fine-tuning
- **Multi-agent consensus** provides verification but at higher token cost

## Installation

```bash
# Clone the repository
git clone https://github.com/abhinav30219/MedLogicTrace.git
cd MedLogicTrace

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

```bash
# 1. Mathematical Pretraining
python run_logictrace_math_pretraining.py

# 2. Medical Transfer Learning
python run_medical_transfer_learning.py

# 3. Multi-Agent System Evaluation
python run_final_evaluation.py
```

### Individual Components

```bash
# Baseline Evaluation
python run_experiments.py

# GRPO Training
python run_custom_grpo_training.py
python run_parallel_custom_grpo.py

# LogicTrace Training
python run_logictrace_math_pretraining.py

# Result Visualization
python create_milestone_plots.py
python create_grpo_comparison_plots.py
```

## Project Structure

```
MedLogicTrace/
├── src/                                    # Core implementation
│   ├── config.py                          # Configuration management
│   ├── data_utils.py                      # Medical dataset utilities
│   ├── math_data_utils.py                 # Mathematical dataset utilities
│   ├── grpo_trainer.py                    # Base GRPO implementation
│   ├── logictrace_optimizer.py            # LogicTrace optimization framework
│   ├── logictrace_grpo_trainer.py         # LogicTrace-enhanced GRPO
│   ├── multi_agent_medical_rl.py          # Multi-agent medical system
│   └── medical_evaluator.py               # Medical task evaluation
├── run_experiments.py                      # Baseline evaluation
├── run_custom_grpo_training.py            # GRPO training
├── run_logictrace_math_pretraining.py     # Mathematical pretraining
├── run_medical_transfer_learning.py       # Medical transfer learning
├── run_final_evaluation.py                # Comprehensive evaluation
├── results/                               # Evaluation results and plots
│   ├── baseline_results.csv               # Baseline performance
│   ├── grpo_evaluation_results.json       # GRPO results
│   └── *.png                              # Visualization plots
└── models/                                # Trained model checkpoints
```

## Key Components

### LogicTrace Optimizer (`src/logictrace_optimizer.py`)
- Structure-aware loss function preserving logical steps
- Step importance weighting (critical vs supporting steps)
- Dynamic length penalties based on problem complexity
- Multi-component reward: accuracy + structure + length

### LogicTrace-Enhanced GRPO (`src/logictrace_grpo_trainer.py`)
- Integration of LogicTrace with GRPO training
- DAPO enhancements: Clip-Higher, dynamic sampling, token-level rewards
- Progressive curriculum learning support
- Checkpoint saving with full configuration

### Multi-Agent Medical System (`src/multi_agent_medical_rl.py`)
- Specialized agents: Diagnostic, Treatment, Verification, Efficiency
- Consensus building mechanisms
- Confidence-weighted voting
- Quality evaluation metrics

### Mathematical Transfer Learning (`src/math_data_utils.py`)
- GSM8K dataset support
- Synthetic mathematical problem generation
- Progressive transfer curriculum: math → medical
- Complexity estimation for dynamic penalties

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

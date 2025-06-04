#!/bin/bash
# Launch script for MedLogicTrace training with Accelerate multi-GPU support

echo "=========================================="
echo "MedLogicTrace Multi-GPU Training Launch"
echo "=========================================="

# Check if token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN environment variable not set!"
    echo "Please run: export HUGGINGFACE_TOKEN='your_token_here'"
    exit 1
fi

# Set CUDA environment for stability
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Create necessary directories
mkdir -p checkpoints
mkdir -p tensorboard_logs
mkdir -p results
mkdir -p hf_upload

# Install Accelerate if not already installed
echo "Checking for Accelerate..."
pip install -q accelerate

# Configure Accelerate for multi-GPU
echo "Configuring Accelerate for $NUM_GPUS GPUs..."
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Launch training with Accelerate
echo ""
echo "Starting MedLogicTrace training on $NUM_GPUS GPUs..."
echo "- Math dataset: Bespoke-Stratos-17k (1k samples)"
echo "- Medical dataset: MedMCQA + PubMedQA (5k samples)"
echo "- Model: Qwen2.5-0.5B-Instruct"
echo "- HuggingFace user: abhinav302019"
echo ""

# Run with Accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes $NUM_GPUS \
    run_complete_training_with_evaluation.py 2>&1 | tee training_output.log

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Check your HuggingFace repos:"
    echo "- https://huggingface.co/abhinav302019/medlogictrace-stratos-math"
    echo "- https://huggingface.co/abhinav302019/medlogictrace-stratos-final"
    echo ""
    echo "TensorBoard logs: tensorboard_logs/"
    echo "Results: results/"
    echo ""
    echo "Training used $NUM_GPUS GPUs with Accelerate"
else
    echo ""
    echo "=========================================="
    echo "Training failed! Check training_output.log for details"
    echo "=========================================="
fi

# Cleanup
rm -f accelerate_config.yaml

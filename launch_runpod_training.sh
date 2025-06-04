#!/bin/bash
# RunPod launch script for MedLogicTrace training with HuggingFace integration

echo "=========================================="
echo "MedLogicTrace RunPod Training Launch"
echo "=========================================="

# IMPORTANT: Set your HuggingFace token as an environment variable
# export HUGGINGFACE_TOKEN="your_token_here"

# Check if token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN environment variable not set!"
    echo "Please run: export HUGGINGFACE_TOKEN='your_token_here'"
    exit 1
fi

# Set CUDA environment for stability
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create necessary directories
mkdir -p checkpoints
mkdir -p tensorboard_logs
mkdir -p results
mkdir -p hf_upload

# Install required packages if not already installed
echo "Checking dependencies..."
pip install -q huggingface-hub matplotlib tensorboard

# Launch training
echo "Starting MedLogicTrace training..."
echo "- Math dataset: Bespoke-Stratos-17k (1k samples)"
echo "- Medical dataset: MedMCQA + PubMedQA (5k samples)"
echo "- Model: Qwen2.5-0.5B-Instruct"
echo "- HuggingFace user: abhinav302019"

# Run the comprehensive training script with multi-agent evaluation
python run_complete_training_with_evaluation.py 2>&1 | tee training_output.log

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
else
    echo ""
    echo "=========================================="
    echo "Training failed! Check training_output.log for details"
    echo "=========================================="
fi

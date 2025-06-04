#!/bin/bash
# Secure credential setup for MedLogicTrace training

echo "=========================================="
echo "MedLogicTrace Credential Setup"
echo "=========================================="

# Function to safely read password
read_token() {
    echo -n "Enter your HuggingFace token: "
    read -s token
    echo
    export HUGGINGFACE_TOKEN="$token"
}

# Check if token is already set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "HuggingFace token not found in environment."
    read_token
else
    echo "HuggingFace token found in environment."
    echo -n "Use existing token? (y/n): "
    read answer
    if [ "$answer" != "y" ]; then
        read_token
    fi
fi

# Verify token format
if [[ ! "$HUGGINGFACE_TOKEN" =~ ^hf_ ]]; then
    echo "WARNING: Token doesn't start with 'hf_'. Please verify it's correct."
fi

# Test token by checking if we can access HuggingFace
echo "Testing HuggingFace connection..."
python -c "
from huggingface_hub import HfApi
import os
try:
    api = HfApi(token=os.environ.get('HUGGINGFACE_TOKEN'))
    user = api.whoami()
    print(f'✓ Successfully authenticated as: {user[\"name\"]}')
except Exception as e:
    print(f'✗ Authentication failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Credentials configured successfully!"
    echo ""
    echo "You can now run: ./launch_runpod_training.sh"
else
    echo ""
    echo "✗ Failed to authenticate with HuggingFace"
    echo "Please check your token and try again"
fi

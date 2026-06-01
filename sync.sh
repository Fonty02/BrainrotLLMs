#!/bin/bash
set -euo pipefail

cd /lustrehome/fonty/CliCIT2026/BrainrotLLMs
UV=/lustrehome/fonty/.local/bin/uv

# Remove any existing torch to avoid conflicts
$UV remove torch torchvision 2>/dev/null || true

# Install torch + torchvision from PyTorch CUDA 12.1 index (matching cu121 config)
$UV pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Ensure python-dotenv is installed for .env loading
$UV add python-dotenv

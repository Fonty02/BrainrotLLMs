#!/bin/bash
# HTCondor wrapper script for judge.py
# Launched via HTC with all arguments provided — no defaults.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Activate virtualenv if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_LOGS="-all"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NCCL_DEBUG=WARN
ulimit -n 65536 2>/dev/null || true

echo "========================================"
echo "Brainrot Judge"
echo "Input CSV:    (from --input_csv)"
echo "Output CSV:   (from --output_csv)"
echo "========================================"
echo "Command: judge.py" "$@"
echo "========================================"

time python -u -W ignore judge.py "$@"

echo "========================================"
echo "Completed successfully"
echo "========================================"

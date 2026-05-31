#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

MODEL="$1"
TECHNIQUE="$2"
LAYER_PCT="$3"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
OUTPUT_CSV="$PROJECT_ROOT/steering_results/${MODEL_SHORT}_${TECHNIQUE}_L${LAYER_PCT}.csv"

export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_LOGS="-all"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NCCL_DEBUG=WARN

ulimit -n 65536 2>/dev/null || true

echo "========================================"
echo "Brainrot Steering Experiment"
echo "Model:     $MODEL"
echo "Technique: $TECHNIQUE"
echo "Layer PCT: $LAYER_PCT"
echo "Output:    $OUTPUT_CSV"
echo "========================================"

time python -u -W ignore steering_experiment.py \
    --model "$MODEL" \
    --technique "$TECHNIQUE" \
    --layer_pct "$LAYER_PCT" \
    --output_csv "$OUTPUT_CSV" \
    --device cuda \
    --max_pairs 2000 \
    --coefficients 5,10,25,-5,-10,-25 \
    --hf_cache_dir /lustrehome/fonty/huggingface_cache

echo "========================================"
echo "Completed: $MODEL | $TECHNIQUE | L$LAYER_PCT"
echo "========================================"

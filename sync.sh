#!/bin/bash
set -euo pipefail

cd /lustrehome/fonty/CliCIT2026/BrainrotLLMs
UV=/lustrehome/fonty/.local/bin/uv

$UV sync

# Download models and dataset to HuggingFace cache
HF_CACHE=/lustrehome/fonty/huggingface_cache
HF_TOKEN="${HF_TOKEN:-}"

$UV run python - <<'EOF'
import os, sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

MODELS = [
    "google/gemma-4-E2B-it",
]

cache_dir = os.environ.get("HF_CACHE")
token = os.environ.get("HF_TOKEN") or None

print("Downloading dataset: shvn22k/brainrot-dataset", flush=True)
load_dataset("shvn22k/brainrot-dataset", split="train", cache_dir=cache_dir)
print("Dataset OK", flush=True)

failed = []
for model_id in MODELS:
    print(f"\n{'='*60}", flush=True)
    print(f"Downloading: {model_id}", flush=True)
    is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()
    try:
        AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir=cache_dir, device_map="cpu")
        print(f"  Model weights OK", flush=True)
        if is_gemma4:
            AutoProcessor.from_pretrained(model_id, token=token, cache_dir=cache_dir)
            print(f"  Processor OK", flush=True)
        else:
            AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache_dir)
            print(f"  Tokenizer OK", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        failed.append(model_id)

if failed:
    print(f"\nFAILED: {failed}", flush=True)
    sys.exit(1)
print("\nAll models downloaded successfully.", flush=True)
EOF

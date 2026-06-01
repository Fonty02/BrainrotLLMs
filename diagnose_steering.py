"""
diagnose_steering.py — Fast local sweep to find the steering "sweet spot".

The full HTCondor grid is expensive and was previously run with a mis-scaled,
fixed coefficient set. Before committing the cluster again, use this to sweep a
FINE coefficient ladder for ONE (model, technique, layer) on a handful of probe
questions, and eyeball where the model goes from "no effect" → "brainrot" →
"degenerate". Prints cheap auto-metrics (brainrot-keyword hits, repetition,
distinct-token ratio) plus a snippet of each response, so you don't need the
full judge to read the frontier.

Usage:
    python diagnose_steering.py --model meta-llama/Llama-3.1-8B-Instruct \
        --technique dom --layer_pct 50 --hf_token hf_xxx
    python diagnose_steering.py --model google/gemma-4-E2B-it --technique aas \
        --layer_pct 75 --coefficients 5,10,15,20,25,30,40 --layer_band 1
"""

import argparse
import re

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from steering_experiment import (
    compute_steering_vector,
    detect_layers,
    extract_dataset_activations,
    generate_steered_response,
    set_seed,
)

load_dotenv()

# A few probe questions spanning categories (cheap to run).
PROBE_QUESTIONS = [
    "What should I cook for dinner tonight?",
    "How do I deal with stress at work?",
    "Who do you think is the best musician of all time?",
    "What's the most interesting thing about black holes?",
    "What caused the French Revolution?",
    "If you could have any superpower, what would you pick?",
]

# Slang markers (subset of the judge's list) for a quick brainrot signal.
BRAINROT_KEYWORDS = [
    "rizz", "no cap", "slay", "fr fr", "fr", "bussin", "based", "lowkey",
    "highkey", "npc", "main character", "understood the assignment",
    "ate and left no crumbs", "it's giving", "rent free", "caught in 4k",
    "bestie", "fam", "bro", "vibe", "vibes", "slaps", "sus", "yeet", "bet",
    "glow up", "mid", "cap", "deadass", "sigma", "gyatt", "skibidi", "ohio",
    "delulu", "ick", "tea", "sheesh", "goated", "w ", "l ", "fanum",
]


def brainrot_hits(text):
    t = text.lower()
    return sum(1 for kw in BRAINROT_KEYWORDS if kw in t)


def max_token_run(text):
    toks = text.split()
    if len(toks) < 2:
        return 0
    best = cur = 1
    for i in range(1, len(toks)):
        cur = cur + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, cur)
    return best


def distinct_ratio(text):
    toks = text.split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def looks_degenerate(text):
    """Same spirit as analysis.is_degenerate, for a quick flag."""
    t = text.strip()
    if len(t.split()) < 5:
        return True
    if max_token_run(t) >= 5:
        return True
    alpha = re.findall(r"[a-zA-Z]", t)
    if alpha and len(set(c.lower() for c in alpha)) < 8:
        return True
    return False


def load_model_and_tokenizer(model_id, hf_token, hf_cache_dir):
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=load_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=hf_token,
        cache_dir=hf_cache_dir,
    )
    model.eval()
    is_gemma4 = "gemma-4" in model_id.lower() or "gemma4" in model_id.lower()
    if is_gemma4:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token, cache_dir=hf_cache_dir)
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=hf_cache_dir)
    return model, tokenizer, processor


def main():
    p = argparse.ArgumentParser(description="Fast local steering sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--technique", required=True, choices=["dom", "pca", "aas"])
    p.add_argument("--layer_pct", required=True, type=int)
    p.add_argument("--layer_band", type=int, default=0)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--max_pairs", type=int, default=500,
                   help="Fewer pairs than the full run — vectors stabilise fast.")
    p.add_argument("--coefficients", default=None,
                   help="Fine ladder. Default: dom/pca 0.5..6, aas 5..50 degrees.")
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--snippet", type=int, default=160, help="Chars of response to print.")
    args = p.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coefficients is not None:
        coeffs = [float(c.strip()) for c in args.coefficients.split(",")]
    elif args.technique == "aas":
        coeffs = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    else:
        coeffs = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6]

    print(f"Loading dataset...")
    ds = load_dataset("shvn22k/brainrot-dataset", split="train", cache_dir=args.hf_cache_dir)
    print(f"Loading model: {args.model}")
    model, tokenizer, processor = load_model_and_tokenizer(args.model, args.hf_token, args.hf_cache_dir)

    layers, num_layers = detect_layers(model)
    center_idx = max(0, min(int(num_layers * args.layer_pct / 100), num_layers - 1))
    band = max(0, args.layer_band)
    band_idxs = [i for i in range(center_idx - band, center_idx + band + 1) if 0 <= i < num_layers]
    layer_indices = {f"L{i}": i for i in band_idxs}
    print(f"{num_layers} layers; center={center_idx}; steering layers={band_idxs}")

    pos_acts, neg_acts, n_pairs = extract_dataset_activations(
        model, tokenizer, processor, ds, layers, layer_indices, args.max_pairs, device
    )

    steer_specs = []
    center_norm = float("nan")
    for i in band_idxs:
        vec, nat = compute_steering_vector(args.technique, pos_acts[f"L{i}"], neg_acts[f"L{i}"])
        steer_specs.append((i, vec))
        if i == center_idx:
            center_norm = nat
    print(f"Pairs used: {n_pairs}; natural style-shift norm @center = {center_norm:.3f}")

    if args.do_sample:
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                          temperature=args.temperature, top_p=args.top_p,
                          repetition_penalty=args.repetition_penalty)
    else:
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                          repetition_penalty=args.repetition_penalty)

    unit = "deg" if args.technique == "aas" else "x"
    print("\n" + "=" * 78)
    print(f"SWEEP: {args.model} | {args.technique} | L{args.layer_pct}% band={band}")
    print("  metrics averaged over probe questions:")
    print("  kw=brainrot-keyword hits | run=max repeated-token run | dist=distinct ratio")
    print("  degen=fraction flagged degenerate (want LOW)")
    print("=" * 78)
    header = f"{'coeff':>8} | {'kw':>5} | {'run':>5} | {'dist':>5} | {'degen':>6} | sample"
    print(header)
    print("-" * 78)

    for coeff in coeffs:
        kws, runs, dists, degens = [], [], [], []
        first_snippet = ""
        for qi, q in enumerate(PROBE_QUESTIONS):
            try:
                resp = generate_steered_response(
                    model, tokenizer, processor, layers, steer_specs,
                    q, args.technique, coeff, gen_kwargs,
                )
            except Exception as e:
                resp = f"ERROR: {e}"
            kws.append(brainrot_hits(resp))
            runs.append(max_token_run(resp))
            dists.append(distinct_ratio(resp))
            degens.append(1.0 if looks_degenerate(resp) else 0.0)
            if qi == 0:
                first_snippet = resp.replace("\n", " ").strip()[:args.snippet]
        n = len(PROBE_QUESTIONS)
        print(f"{coeff:>6.1f}{unit:>2} | {sum(kws)/n:>5.1f} | {sum(runs)/n:>5.1f} | "
              f"{sum(dists)/n:>5.2f} | {sum(degens)/n:>6.2f} | {first_snippet}")

    print("-" * 78)
    print("Read the table: pick the coeff with rising kw / stable dist / low degen.")
    print("That is the sweet spot to center the cluster grid around.")


if __name__ == "__main__":
    main()

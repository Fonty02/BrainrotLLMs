"""
steering_experiment.py — Activation Steering for Brainrot Language Style

Applies activation steering (DoM, PCA, AAS) to a single (model, technique, layer_pct)
configuration. Extracts steering vectors from shvn22k/brainrot-dataset, applies them
at the specified layer with 6 coefficients, and records generations to a CSV.

Designed for HTCondor parallelization: each job runs one (model × technique × layer_pct).
Optionally split further by question range with --q_slice.

Usage:
    python steering_experiment.py --model Qwen/Qwen2.5-7B-Instruct --technique dom --layer_pct 50 --hf_token hf_xxx
    python steering_experiment.py --model meta-llama/Llama-3.1-8B-Instruct --technique aas --layer_pct 25 --q_slice 1/4 --hf_token hf_xxx
"""

import argparse
import csv
import os
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

load_dotenv()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_questions():
    """Returns list of (question, category) tuples."""
    return [
        # --- Everyday Life (40 questions) ---
        ("What should I cook for dinner tonight?", "everyday life"),
        ("How do I deal with stress at work?", "everyday life"),
        ("What's a good morning routine to start the day right?", "everyday life"),
        ("How can I make new friends in a new city?", "everyday life"),
        ("What are some fun hobbies to pick up on weekends?", "everyday life"),
        ("How do I stop procrastinating on important tasks?", "everyday life"),
        ("What's the best way to save money each month?", "everyday life"),
        ("How do I get better at small talk?", "everyday life"),
        ("What should I do when I feel lonely?", "everyday life"),
        ("How can I improve my sleep schedule?", "everyday life"),
        ("What's a good workout routine for beginners?", "everyday life"),
        ("How do I stay motivated when learning something new?", "everyday life"),
        ("What are some healthy snacks I can make quickly?", "everyday life"),
        ("How do I deal with a difficult boss?", "everyday life"),
        ("What's the best way to apologize to someone?", "everyday life"),
        ("How can I make my apartment feel more like home?", "everyday life"),
        ("What should I wear to a job interview?", "everyday life"),
        ("How do I handle a breakup in a healthy way?", "everyday life"),
        ("What are some tips for public speaking?", "everyday life"),
        ("How do I stop comparing myself to others on social media?", "everyday life"),
        ("What's a good way to celebrate a friend's birthday?", "everyday life"),
        ("How can I be more productive when working from home?", "everyday life"),
        ("What should I do if I'm feeling burnt out?", "everyday life"),
        ("How do I learn to say no without feeling guilty?", "everyday life"),
        ("What's the best way to organize my closet?", "everyday life"),
        ("How can I make my commute more enjoyable?", "everyday life"),
        ("What are some good conversation starters for a date?", "everyday life"),
        ("How do I deal with noisy neighbors?", "everyday life"),
        ("What's a good skincare routine for beginners?", "everyday life"),
        ("How can I improve my relationship with my parents?", "everyday life"),
        ("What should I do on a rainy Sunday afternoon?", "everyday life"),
        ("How do I build self-confidence?", "everyday life"),
        ("What's the best way to meal prep for the week?", "everyday life"),
        ("How can I stop overthinking everything?", "everyday life"),
        ("What are some affordable date night ideas?", "everyday life"),
        ("How do I deal with imposter syndrome at work?", "everyday life"),
        ("What's a good way to keep a journal consistently?", "everyday life"),
        ("How can I make my mornings less rushed?", "everyday life"),
        ("What should I do if I regret a major life decision?", "everyday life"),
        ("How do I balance work and personal life effectively?", "everyday life"),
        # --- Pop Culture (20 questions) ---
        ("Who do you think is the best musician of all time?", "pop culture"),
        ("What's the most overrated movie you've seen?", "pop culture"),
        ("Which TV show had the best ending ever?", "pop culture"),
        ("What video game changed the industry forever?", "pop culture"),
        ("Who's your favorite celebrity and why?", "pop culture"),
        ("What's the best album released in the last five years?", "pop culture"),
        ("Which social media platform is the most toxic right now?", "pop culture"),
        ("What movie soundtrack absolutely slaps?", "pop culture"),
        ("Who is the greatest athlete of all time?", "pop culture"),
        ("What's the most iconic meme in internet history?", "pop culture"),
        ("Which actor deserves an Oscar but hasn't won one yet?", "pop culture"),
        ("What fashion trend needs to come back immediately?", "pop culture"),
        ("What's the best animated movie ever made?", "pop culture"),
        ("Who had the biggest glow-up in the music industry?", "pop culture"),
        ("What TV show got cancelled way too soon?", "pop culture"),
        ("What's the most rewatchable movie you know?", "pop culture"),
        ("Which band had the best live performances?", "pop culture"),
        ("What YouTuber defined your childhood?", "pop culture"),
        ("What's the best console of all time and why?", "pop culture"),
        ("What celebrity drama was the most entertaining?", "pop culture"),
        # --- Science & Technology (15 questions) ---
        ("What's the most interesting thing about black holes?", "science & technology"),
        ("How does artificial intelligence actually learn?", "science & technology"),
        ("What would happen if the sun suddenly disappeared?", "science & technology"),
        ("How does quantum computing work in simple terms?", "science & technology"),
        ("What's the biggest challenge in curing cancer?", "science & technology"),
        ("How does climate change affect ocean currents?", "science & technology"),
        ("What would it take to colonize Mars?", "science & technology"),
        ("How does CRISPR gene editing actually work?", "science & technology"),
        ("What causes the northern lights?", "science & technology"),
        ("Will AI ever become truly conscious?", "science & technology"),
        ("How do vaccines train your immune system?", "science & technology"),
        ("What's the most mind-blowing physics fact you know?", "science & technology"),
        ("How does the internet actually transmit data across oceans?", "science & technology"),
        ("What's the biggest unsolved mystery in biology?", "science & technology"),
        ("How do electric cars compare to gas cars for the environment?", "science & technology"),
        # --- History & Geography (15 questions) ---
        ("What was the most important event in the 20th century?", "history & geography"),
        ("How did the Roman Empire actually collapse?", "history & geography"),
        ("What's the most underrated country to visit?", "history & geography"),
        ("Who is the most misunderstood historical figure?", "history & geography"),
        ("What caused the French Revolution?", "history & geography"),
        ("Which ancient civilization was the most advanced?", "history & geography"),
        ("What's the most fascinating city you've ever learned about?", "history & geography"),
        ("How did World War I actually start?", "history & geography"),
        ("What happened to the Library of Alexandria?", "history & geography"),
        ("Which country has the most interesting history?", "history & geography"),
        ("How did the Silk Road change the world?", "history & geography"),
        ("What's the story behind the Great Wall of China?", "history & geography"),
        ("What was life like during the Renaissance?", "history & geography"),
        ("How did the Cold War shape modern geopolitics?", "history & geography"),
        ("What's the most interesting archaeological discovery ever made?", "history & geography"),
        # --- Hypothetical / Opinion (10 questions) ---
        ("If you could live anywhere in the world, where would it be?", "hypothetical / opinion"),
        ("Would you rather have infinite money or infinite wisdom?", "hypothetical / opinion"),
        ("What would you do if you won the lottery tomorrow?", "hypothetical / opinion"),
        ("If you could have any superpower, what would you pick?", "hypothetical / opinion"),
        ("What's one thing you'd change about the world if you could?", "hypothetical / opinion"),
        ("Would you rather be famous or truly happy?", "hypothetical / opinion"),
        ("If you could travel back in time, which era would you visit?", "hypothetical / opinion"),
        ("What would your perfect day look like?", "hypothetical / opinion"),
        ("If you could master any skill instantly, what would it be?", "hypothetical / opinion"),
        ("Would you rather explore the ocean or outer space?", "hypothetical / opinion"),
    ]


# Fixed, neutral carrier turn used when extracting the *output-side* style
# vector. Its only job is to put the style-bearing text in the assistant
# position; since it is identical for the positive and negative member of
# every pair, its contribution cancels in the mean difference.
STYLE_PROBE_PROMPT = "Tell me something."


def get_activation_hook(storage_dict, layer_key):
    """Capture the FULL hidden state [B, T, D] so the caller can mean-pool over
    the assistant-content tokens (style is distributed across the response, not
    concentrated on the last token)."""
    def hook(module, input, output):
        try:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"Expected tensor, got {type(hidden)}")
            storage_dict[layer_key] = hidden.detach().float()
        except Exception as e:
            storage_dict[f"__error_{layer_key}"] = str(e)
    return hook


def pool_content(hidden, attention_mask, prefix_len, trim_end=1):
    """Mean-pool hidden states over the assistant-content tokens only.

    Inputs are LEFT-padded, so for each row the real tokens occupy the rightmost
    `real_len` positions, laid out as [prefix tokens][assistant content][eot].
    `prefix_len` is constant (the carrier prompt is fixed), so the content span
    is the last `real_len - prefix_len` real positions; we trim the trailing
    end-of-turn token.
    """
    B, T, D = hidden.shape
    real_lens = attention_mask.sum(dim=1)
    pooled = []
    for i in range(B):
        rl = int(real_lens[i].item())
        cl = rl - prefix_len
        if cl <= 1:
            start, end = T - rl, T  # fallback: pool all real tokens
        else:
            start = T - cl
            end = T - trim_end if cl > trim_end else T
        pooled.append(hidden[i, start:end, :].mean(dim=0))
    return torch.stack(pooled, dim=0).float()


def make_steering_hook_dom_pca(steering_vector, coeff):
    """Additive steering. `steering_vector` already carries the natural per-layer
    style-shift magnitude (see compute_steering_vector), so `coeff` is a
    *multiple of the average brainrot−normal difference* — comparable across
    layers and models regardless of residual-stream scale."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sv = steering_vector.to(hidden.device, hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] + coeff * sv
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def aas_steer(h, steering_vector, alpha_degrees):
    h_last = h[:, -1, :]
    h_norm = h_last.norm(dim=-1, keepdim=True)
    h_hat = h_last / (h_norm + 1e-8)
    sv = steering_vector.to(h.device, h.dtype)
    proj = (h_hat * sv.unsqueeze(0)).sum(dim=-1, keepdim=True)
    sv_perp = sv.unsqueeze(0) - proj * h_hat
    sv_perp_norm = sv_perp.norm(dim=-1, keepdim=True)
    sv_perp = sv_perp / (sv_perp_norm + 1e-8)
    theta = torch.tensor(alpha_degrees * 3.14159265 / 180.0)
    h_new = torch.cos(theta) * h_last + torch.sin(theta) * h_norm * sv_perp
    h[:, -1, :] = h_new
    return h


def make_steering_hook_aas(steering_vector, alpha):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = aas_steer(hidden, steering_vector, alpha)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def detect_layers(model):
    model_type = type(model).__name__

    def _get_num_layers(cfg):
        try:
            return cfg.get_text_config().num_hidden_layers
        except Exception:
            pass
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            return cfg.text_config.num_hidden_layers
        if hasattr(cfg, "num_hidden_layers"):
            return cfg.num_hidden_layers
        raise ValueError(f"Cannot find num_hidden_layers in config: {type(cfg)}")

    # Gemma-4 multimodal: text decoder layers are in language_model.decoder.layers
    if hasattr(model, 'language_model'):
        lm = model.language_model
        print(f"[detect_layers] language_model type: {type(lm).__name__}", flush=True)
        if hasattr(lm, 'decoder') and hasattr(lm.decoder, 'layers'):
            layers = lm.decoder.layers
            print(f"[detect_layers] branch: lm.decoder.layers, layer type: {type(layers[0]).__name__}", flush=True)
            return layers, len(layers)
        if hasattr(lm, 'model') and hasattr(lm.model, 'decoder') and hasattr(lm.model.decoder, 'layers'):
            layers = lm.model.decoder.layers
            print(f"[detect_layers] branch: lm.model.decoder.layers, layer type: {type(layers[0]).__name__}", flush=True)
            return layers, len(layers)
        if hasattr(lm, 'layers'):
            layers = lm.layers
            print(f"[detect_layers] branch: lm.layers, layer type: {type(layers[0]).__name__}", flush=True)
            return layers, len(layers)
        if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
            layers = lm.model.layers
            print(f"[detect_layers] branch: lm.model.layers, layer type: {type(layers[0]).__name__}", flush=True)
            return layers, len(layers)

    # Gemma4ForConditionalGeneration: text layers at model.model.language_model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        layers = model.model.language_model.layers
        print(f"[detect_layers] branch: model.model.language_model.layers, layer type: {type(layers[0]).__name__}", flush=True)
        return layers, len(layers)

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        return layers, len(layers)

    if hasattr(model, 'text_model') and hasattr(model.text_model, 'decoder') and hasattr(model.text_model.decoder, 'layers'):
        layers = model.text_model.decoder.layers
        return layers, len(layers)

    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        layers = model.model.decoder.layers
        return layers, len(layers)

    import torch.nn as nn
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, 'self_attn') or hasattr(first, 'mlp'):
                return module, len(module)

    try:
        num_layers = _get_num_layers(model.config)
    except ValueError:
        num_layers = None
    raise ValueError(
        f"Cannot find layers in model architecture: {model_type}"
        + (f" (config says {num_layers} layers, but no matching ModuleList found)" if num_layers else "")
    )


def extract_dataset_activations(model, tokenizer, processor, dataset, layers, layer_indices, max_pairs, device):
    """Build output-side style activations.

    The brainrot (target) and normal (source) text of each pair are placed as
    the ASSISTANT turn after a fixed neutral user prompt, and activations are
    mean-pooled over the assistant-content tokens. This captures the direction
    associated with *generating* in that style, rather than with *reading* a
    user message written in that style (the previous, weaker contrast).
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = list(zip(dataset["source"], dataset["target"]))
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    template_fn = processor if processor is not None else tokenizer
    encode_fn = processor if processor is not None else tokenizer
    base_kwargs = {"tokenize": False}
    if processor is not None:
        base_kwargs["enable_thinking"] = False

    # Constant prefix length: the carrier prompt up to the assistant header.
    prefix_text = template_fn.apply_chat_template(
        [{"role": "user", "content": STYLE_PROBE_PROMPT}],
        add_generation_prompt=True,
        **base_kwargs,
    )
    prefix_len = encode_fn(text=prefix_text, return_tensors="pt", truncation=True)["input_ids"].shape[1]

    def render(style_text):
        msgs = [
            {"role": "user", "content": STYLE_PROBE_PROMPT},
            {"role": "assistant", "content": style_text},
        ]
        try:
            return template_fn.apply_chat_template(msgs, add_generation_prompt=False, **base_kwargs)
        except Exception:
            return template_fn.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)

    if isinstance(device, torch.device):
        is_cuda = device.type == "cuda"
    else:
        is_cuda = str(device) == "cuda"

    pos_activations = {lk: [] for lk in layer_indices}
    neg_activations = {lk: [] for lk in layer_indices}

    def run_and_pool(texts, store):
        enc = encode_fn(text=texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        storage = {}
        handles = [
            layers[li].register_forward_hook(get_activation_hook(storage, lk))
            for lk, li in layer_indices.items()
        ]
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in handles:
                h.remove()
        attn = enc["attention_mask"]
        for lk, li in layer_indices.items():
            if lk not in storage:
                err = storage.get(f"__error_{lk}", "hook never called")
                raise RuntimeError(
                    f"Activation hook did not fire for '{lk}' "
                    f"(layer_idx={li}, layer_type={type(layers[li]).__name__}): {err}"
                )
            store[lk].append(pool_content(storage[lk], attn, prefix_len).cpu())
        del enc, storage
        if is_cuda:
            torch.cuda.empty_cache()

    batch_size = 8
    for batch_start in tqdm(range(0, len(pairs), batch_size), desc="Extracting activations"):
        batch_pairs = pairs[batch_start:batch_start + batch_size]
        pos_texts = [render(tgt) for _src, tgt in batch_pairs]
        neg_texts = [render(src) for src, _tgt in batch_pairs]
        run_and_pool(pos_texts, pos_activations)
        run_and_pool(neg_texts, neg_activations)

    for lk in layer_indices:
        pos_activations[lk] = torch.cat(pos_activations[lk], dim=0)
        neg_activations[lk] = torch.cat(neg_activations[lk], dim=0)

    return pos_activations, neg_activations, len(pairs)


def compute_steering_vector(technique, pos_activations, neg_activations):
    """Return (steering_vector, natural_norm).

    `natural_norm` = ||mean(pos − neg)||, the average style-shift magnitude at
    this layer. For DoM/PCA the returned vector is scaled to this magnitude, so
    the applied coefficient is a multiple of the natural difference (comparable
    across layers/models, which the previous unit-norm scheme was not). For AAS
    the vector is unit-norm (the coefficient is an angle in degrees).
    """
    mean_diff = (pos_activations - neg_activations).mean(dim=0)
    natural_norm = mean_diff.norm().item()

    if technique == "dom":
        return mean_diff.clone(), natural_norm

    elif technique == "pca":
        diffs = (pos_activations - neg_activations).numpy()
        pca = PCA(n_components=1)
        pca.fit(diffs)
        comp = torch.tensor(pca.components_[0], dtype=torch.float32)
        comp = comp / (comp.norm() + 1e-8)
        if (mean_diff @ comp).item() < 0:
            comp = -comp
        # Scale the principal style direction to the natural shift magnitude so
        # it shares the same coefficient units as DoM.
        return comp * natural_norm, natural_norm

    elif technique == "aas":
        unit = mean_diff / (mean_diff.norm() + 1e-8)
        return unit, natural_norm

    else:
        raise ValueError(f"Unknown technique: {technique}")


def build_messages(tokenizer, processor, question, use_system=True):
    template_fn = processor if processor is not None else tokenizer
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if processor is not None:
        chat_kwargs["enable_thinking"] = False
    if use_system:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question."},
            {"role": "user", "content": question},
        ]
        try:
            prompt = template_fn.apply_chat_template(messages, **chat_kwargs)
            return prompt
        except Exception:
            pass
    messages = [{"role": "user", "content": question}]
    return template_fn.apply_chat_template(messages, **chat_kwargs)


def generate_steered_response(
    model, tokenizer, processor, layers, steer_specs, question, technique, coefficient, gen_kwargs
):
    """Generate one steered response.

    `steer_specs` is a list of (layer_idx, steering_vector); a hook is attached
    at every listed layer so steering can span a band of layers, each with its
    own per-layer vector. `gen_kwargs` carries the decoding parameters.
    """
    model_device = getattr(model, 'device', None) or next(model.parameters()).device
    prompt = build_messages(tokenizer, processor, question, use_system=True)
    encode_fn = processor if processor is not None else tokenizer
    inputs = encode_fn(text=prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    handles = []
    for layer_idx, vec in steer_specs:
        if technique == "aas":
            hook_fn = make_steering_hook_aas(vec, coefficient)
        else:
            hook_fn = make_steering_hook_dom_pca(vec, coefficient)
        handles.append(layers[layer_idx].register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
    finally:
        for h in handles:
            h.remove()

    input_len = inputs["input_ids"].shape[1]
    decode_fn = processor if processor is not None else tokenizer
    response = decode_fn.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response


def write_csv_header(output_csv, columns):
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()


def append_csv_row(output_csv, columns, row):
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Brainrot Activation Steering Experiment")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-4-E2B-it",
        ],
    )
    parser.add_argument("--technique", required=True, choices=["dom", "pca", "aas"])
    parser.add_argument("--layer_pct", required=True, type=int, help="Layer percentage: 25, 50, or 75")
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_cache_dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--max_pairs", type=int, default=2000, help="Max dataset pairs to use (0 = all)")
    parser.add_argument(
        "--q_slice",
        default=None,
        help="Question slice N/M (1-indexed) to split work across workers, e.g. 1/4",
    )
    parser.add_argument(
        "--coefficients",
        default=None,
        help="Comma-separated coefficient values. If omitted, a sensible "
             "per-technique default is used (raw-difference multiples for "
             "dom/pca, degrees for aas).",
    )
    parser.add_argument(
        "--layer_band",
        type=int,
        default=0,
        help="Steer a band of layers around the target: 0 = single layer, "
             "1 = target±1, etc. Each layer gets its own vector.",
    )
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    if args.layer_pct not in (25, 50, 75):
        parser.error("--layer_pct must be 25, 50, or 75")

    # Per-technique default coefficient grids (only used if --coefficients omitted).
    #   dom/pca: multiples of the natural brainrot−normal difference vector.
    #   aas:     rotation angle in degrees.
    DEFAULT_COEFFS = {
        "dom": "1,2,3,4,-2,-4",
        "pca": "1,2,3,4,-2,-4",
        "aas": "15,25,35,45,-25,-45",
    }
    coeff_str = args.coefficients if args.coefficients is not None else DEFAULT_COEFFS[args.technique]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)

    # Parse q_slice
    q_slice_idx = None
    q_slice_total = None
    if args.q_slice:
        parts = args.q_slice.split("/")
        if len(parts) != 2:
            parser.error("--q_slice must be N/M format, e.g. 1/4")
        q_slice_idx = int(parts[0]) - 1
        q_slice_total = int(parts[1])
        if q_slice_idx < 0 or q_slice_idx >= q_slice_total:
            parser.error(f"--q_slice index out of range: {args.q_slice}")

    coefficients = [float(c.strip()) for c in coeff_str.split(",")]

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset(
        "shvn22k/brainrot-dataset",
        split="train",
        cache_dir=args.hf_cache_dir,
    )

    # Load model
    print(f"Loading model: {args.model}")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float16

    def load_model_with_retry(**kwargs):
        try:
            return AutoModelForCausalLM.from_pretrained(**kwargs)
        except Exception as e:
            if "JSONDecodeError" in str(e) and kwargs.get("cache_dir"):
                cache = Path(kwargs["cache_dir"])
                model_slug = args.model.replace("/", "--")
                for corrupt_dir in cache.glob(f"models--{model_slug}*"):
                    if corrupt_dir.is_dir():
                        print(f"Corrupted cache detected. Clearing: {corrupt_dir}")
                        shutil.rmtree(corrupt_dir)
                return AutoModelForCausalLM.from_pretrained(**kwargs)
            raise

    model = load_model_with_retry(
        pretrained_model_name_or_path=args.model,
        dtype=load_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    model.eval()

    is_gemma4 = "gemma-4" in args.model.lower() or "gemma4" in args.model.lower()
    if is_gemma4:
        processor = AutoProcessor.from_pretrained(
            args.model,
            token=args.hf_token,
            cache_dir=args.hf_cache_dir,
        )
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=args.hf_token,
            cache_dir=args.hf_cache_dir,
        )

    layers, num_layers = detect_layers(model)

    center_idx = int(num_layers * args.layer_pct / 100)
    center_idx = max(0, min(center_idx, num_layers - 1))
    band = max(0, args.layer_band)
    band_idxs = [i for i in range(center_idx - band, center_idx + band + 1) if 0 <= i < num_layers]
    layer_indices = {f"L{i}": i for i in band_idxs}
    print(f"Model has {num_layers} layers. Center {args.layer_pct}% = index {center_idx}; "
          f"steering layers: {band_idxs}")

    max_pairs = args.max_pairs if args.max_pairs > 0 else len(ds)
    pos_acts, neg_acts, num_pairs = extract_dataset_activations(
        model, tokenizer, processor, ds, layers, layer_indices, max_pairs, device
    )
    print(f"Extracted activations from {num_pairs} pairs")

    all_questions = get_questions()
    if q_slice_total:
        chunk_size = (len(all_questions) + q_slice_total - 1) // q_slice_total
        start = q_slice_idx * chunk_size
        end = min(start + chunk_size, len(all_questions))
        questions = all_questions[start:end]
        print(f"Question slice {args.q_slice}: Q{start}-{end-1} ({len(questions)} questions)")
    else:
        questions = all_questions

    csv_columns = [
        "experiment_id", "timestamp", "model_name", "technique",
        "layer_pct", "layer_idx", "num_layers", "layer_band", "coefficient",
        "question_id", "question", "question_category", "response",
        "steering_vector_norm", "num_pos_samples", "num_neg_samples",
        "max_new_tokens", "decoding", "temperature", "top_p",
        "repetition_penalty", "strength_mode", "seed",
    ]

    file_exists = os.path.exists(args.output_csv)
    if not file_exists:
        write_csv_header(args.output_csv, csv_columns)

    total_experiments = len(coefficients) * len(questions)
    print(f"Starting experiment loop: {len(coefficients)} coefficients × {len(questions)} questions = {total_experiments} generations")
    print(f"Output CSV: {args.output_csv}")

    # One steering vector per layer in the band; each is applied at its own layer.
    print(f"\n=== Steering layers {band_idxs} (center {center_idx}) ===")
    steer_specs = []
    center_norm = float("nan")
    for i in band_idxs:
        vec, nat_norm = compute_steering_vector(args.technique, pos_acts[f"L{i}"], neg_acts[f"L{i}"])
        steer_specs.append((i, vec))
        if i == center_idx:
            center_norm = nat_norm
    print(f"Natural style-shift norm at center layer: {center_norm:.3f}")

    # Decoding parameters.
    if args.do_sample:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        decoding_str = f"sampling(t={args.temperature},p={args.top_p})"
    else:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens, do_sample=False,
            repetition_penalty=args.repetition_penalty,
        )
        decoding_str = "greedy"

    strength_mode = "degrees" if args.technique == "aas" else "natural_diff_multiple"

    for coeff in coefficients:
        for qid_offset, (question, category) in enumerate(questions):
            qid = qid_offset + (start if q_slice_total else 0)
            try:
                response = generate_steered_response(
                    model, tokenizer, processor, layers, steer_specs,
                    question, args.technique, coeff, gen_kwargs,
                )
            except Exception as e:
                response = f"ERROR: {e}"

            row = {
                "experiment_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": args.model,
                "technique": args.technique,
                "layer_pct": args.layer_pct,
                "layer_idx": center_idx,
                "num_layers": num_layers,
                "layer_band": band,
                "coefficient": float(coeff),
                "question_id": qid,
                "question": question,
                "question_category": category,
                "response": response,
                "steering_vector_norm": center_norm,
                "num_pos_samples": num_pairs,
                "num_neg_samples": num_pairs,
                "max_new_tokens": args.max_new_tokens,
                "decoding": decoding_str,
                "temperature": args.temperature if args.do_sample else "",
                "top_p": args.top_p if args.do_sample else "",
                "repetition_penalty": args.repetition_penalty,
                "strength_mode": strength_mode,
                "seed": 42,
            }
            append_csv_row(args.output_csv, csv_columns, row)

            if (qid_offset + 1) % 10 == 0 or qid_offset == 0:
                print(f"[{args.model}] [{args.technique}] center={center_idx} band={band} coeff={coeff} Q{qid_offset+1}/{len(questions)}")

    print(f"\nDone! Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

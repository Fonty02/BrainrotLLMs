"""probe_brainrot_knowledge.py — does the model KNOW the term "brainrot"?

Asks each target model, with NO steering, simply to "Generate a brainrot
sentence." and records the raw response. The premise of the paper is that the
older models (Qwen2.5, Llama-3.1) were trained before "brainrot" entered common
use and do not recognise the term, whereas the more recent Gemma does. This
script produces the evidence for that claim.

Writes brainrot_knowledge_probe.txt (UTF-8) with one block per model.

Usage:
    python probe_brainrot_knowledge.py --hf_token hf_xxx
    python probe_brainrot_knowledge.py --hf_token hf_xxx --hf_cache_dir /path/cache
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The three target models studied in the paper.
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-4-E2B-it",
]

# A single, neutral instruction. The point is to see whether the model has any
# notion of what "brainrot" is, unaided by steering or few-shot examples.
PROMPT = "Generate a brainrot sentence."

# A couple of follow-up probes that are useful as corroboration (optional but
# cheap): ask the model to define the term and to give an example.
EXTRA_PROMPTS = [
    "What does the internet slang term \"brainrot\" mean?",
    "Write one sentence in \"brainrot\" style.",
]


def generate(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Probe model knowledge of 'brainrot'")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--output", default="brainrot_knowledge_probe.txt")
    args = parser.parse_args()
    if not args.hf_token:
        parser.error("--hf_token is required (or set the HF_TOKEN environment variable)")

    doc = ["BRAINROT KNOWLEDGE PROBE — does each model recognise the term, unaided?",
           f"Main prompt: {PROMPT!r}\n"]

    for model_id in MODELS:
        print(f"Loading {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=args.hf_token, cache_dir=args.hf_cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=args.hf_token,
            cache_dir=args.hf_cache_dir,
        )
        model.eval()

        doc.append("=" * 70)
        doc.append(model_id)
        doc.append("=" * 70)
        for p in [PROMPT, *EXTRA_PROMPTS]:
            resp = generate(model, tokenizer, p)
            print(f"\n[{model_id}] {p}\n{resp}\n")
            doc.append(f"\n> {p}\n{resp}")
        doc.append("")

        # free GPU memory before the next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(args.output).write_text("\n".join(doc), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

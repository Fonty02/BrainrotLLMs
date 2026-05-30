"""
judge.py — Brainrot Style Classifier for Steering Experiment Results

Reads a CSV produced by steering_experiment.py, evaluates each response using
a judge model, and writes a new CSV with an added `is_brainrot` column.

Supports --row_slice N/M for HTCondor parallelization across multiple workers
processing the same input CSV.

Usage:
    python judge.py --input_csv results.csv --output_csv judged.csv --hf_token hf_xxx
    python judge.py --input_csv results.csv --output_csv judged.csv --row_slice 1/4 --hf_token hf_xxx
"""

import argparse
import csv
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an expert evaluator of internet language styles. 
Your task is to determine whether a given text is written in "brainrot" style or not.

Brainrot style is characterized by:
- Use of Gen-Z / internet slang (e.g., "rizz", "no cap", "slay", "fr fr", "bussin", "based", "lowkey", "NPC", "main character", "understood the assignment", "ate and left no crumbs", "it's giving", "rent free", "caught in 4K", "real", "I can't even", "bestie", "fam", "bro", "W", "L", "era")
- Meme references and internet humor
- Ironic or chaotic sentence structure
- Deliberate misspellings or unconventional capitalization
- Short fragmented sentences mixed with run-ons
- Emoji-like expressiveness even without actual emojis
- References to viral internet culture

Answer with ONLY the single word YES if the text is brainrot style, or NO if it is not. Do not explain."""

COHERENCE_SYSTEM_PROMPT = """You are an expert evaluator of text quality and coherence.
Your task is to determine whether a given response is coherent and logically consistent.

A COHERENT response:
- Follows a logical flow and stays on topic
- Has clear connections between sentences
- Addresses the question that was asked
- Is understandable and makes sense as a whole

An INCOHERENT response:
- Jumps between unrelated ideas or is self-contradictory
- Is garbled, nonsensical, or impossible to follow
- Fails to address the original question
- Contains broken grammar that obscures meaning

Answer with ONLY the single word COHERENT if coherent, or INCOHERENT if not. Do not explain."""


def parse_judge_response(response_text):
    answer = response_text.strip().upper()
    if "YES" in answer:
        return 1
    elif "NO" in answer:
        return 0
    else:
        return -1


def parse_coherence_response(response_text):
    answer = response_text.strip().upper()
    if "COHERENT" in answer:
        return 1
    elif "INCOHERENT" in answer:
        return 0
    else:
        return -1


def build_judge_prompt(question, response, judge_type):
    if judge_type == "brainrot":
        system = SYSTEM_PROMPT
        user = (
            f"Question that was asked: {question}\n\n"
            f"Response to evaluate:\n{response}\n\n"
            f"Is this response written in brainrot style?"
        )
    else:
        system = COHERENCE_SYSTEM_PROMPT
        user = (
            f"Question that was asked: {question}\n\n"
            f"Response to evaluate:\n{response}\n\n"
            f"Is this response coherent and logically consistent?"
        )
    return system, user


def main():
    parser = argparse.ArgumentParser(description="Brainrot Style Judge")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--model_id", default="google/gemma-4-E4B-it")
    parser.add_argument("--judge_type", default="brainrot", choices=["brainrot", "coherence", "brainrot,coherence"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--row_slice",
        default=None,
        help="Row slice N/M (1-indexed), e.g. 1/4",
    )
    args = parser.parse_args()

    row_slice_idx = None
    row_slice_total = None
    if args.row_slice:
        parts = args.row_slice.split("/")
        if len(parts) != 2:
            parser.error("--row_slice must be N/M format, e.g. 1/4")
        row_slice_idx = int(parts[0]) - 1
        row_slice_total = int(parts[1])
        if row_slice_idx < 0 or row_slice_idx >= row_slice_total:
            parser.error(f"--row_slice index out of range: {args.row_slice}")

    print(f"Loading judge model: {args.model_id}")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=args.hf_token, cache_dir=args.hf_cache_dir,
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    judge_model.eval()

    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        input_columns = reader.fieldnames

    total_rows = len(all_rows)

    if row_slice_total:
        chunk_size = (total_rows + row_slice_total - 1) // row_slice_total
        start = row_slice_idx * chunk_size
        end = min(start + chunk_size, total_rows)
        rows = all_rows[start:end]
        print(f"Loaded {total_rows} rows total, processing slice {args.row_slice}: rows {start}-{end-1} ({len(rows)} rows)")
    else:
        rows = all_rows
        print(f"Loaded {total_rows} rows from {args.input_csv}")

    judge_types = [jt.strip() for jt in args.judge_type.split(",")]

    if judge_types == ["brainrot"]:
        output_columns = input_columns + ["is_brainrot", "judge_model", "judge_raw_response"]
    elif judge_types == ["coherence"]:
        output_columns = input_columns + ["is_coherent", "judge_model", "judge_raw_response"]
    else:
        output_columns = input_columns + [
            "is_brainrot", "is_coherent", "judge_model",
            "judge_brainrot_raw", "judge_coherence_raw",
        ]

    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        if not file_exists:
            writer.writeheader()

        batch_size = args.batch_size
        for batch_start in tqdm(range(0, len(rows), batch_size), desc="Judging"):
            batch = rows[batch_start:batch_start + batch_size]

            batch_results = {}

            for jtype in judge_types:
                prompts = []
                for row in batch:
                    system, user = build_judge_prompt(row["question"], row["response"], jtype)
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                    prompt = judge_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)

                inputs = judge_tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = judge_model.generate(
                        **inputs,
                        max_new_tokens=5,
                        temperature=None,
                        do_sample=False,
                    )

                input_lens = inputs["input_ids"].shape[1]
                responses = judge_tokenizer.batch_decode(
                    output_ids[:, input_lens:], skip_special_tokens=True
                )
                batch_results[jtype] = responses

            for i, row in enumerate(batch):
                row["judge_model"] = args.model_id
                if "brainrot" in judge_types:
                    raw_br = batch_results["brainrot"][i]
                    is_brainrot = parse_judge_response(raw_br)
                    row["is_brainrot"] = is_brainrot
                    if len(judge_types) == 2:
                        row["judge_brainrot_raw"] = raw_br
                    else:
                        row["judge_raw_response"] = raw_br
                if "coherence" in judge_types:
                    raw_coh = batch_results["coherence"][i]
                    is_coherent = parse_coherence_response(raw_coh)
                    row["is_coherent"] = is_coherent
                    if len(judge_types) == 2:
                        row["judge_coherence_raw"] = raw_coh
                    else:
                        row["judge_raw_response"] = raw_coh
                writer.writerow(row)

    print(f"Done! Judged results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

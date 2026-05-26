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


def parse_judge_response(response_text):
    answer = response_text.strip().upper()
    if "YES" in answer:
        return 1
    elif "NO" in answer:
        return 0
    else:
        return -1


def main():
    parser = argparse.ArgumentParser(description="Brainrot Style Judge")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--model_id", default="google/gemma-4-E4B-it")
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

    output_columns = input_columns + ["is_brainrot", "judge_model", "judge_raw_response"]

    file_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        if not file_exists:
            writer.writeheader()

        batch_size = args.batch_size
        for batch_start in tqdm(range(0, len(rows), batch_size), desc="Judging"):
            batch = rows[batch_start:batch_start + batch_size]

            prompts = []
            for row in batch:
                user_prompt = f"""Question that was asked: {row['question']}

Response to evaluate:
{row['response']}

Is this response written in brainrot style?"""
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
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

            for row, raw_response in zip(batch, responses):
                is_brainrot = parse_judge_response(raw_response)
                row["is_brainrot"] = is_brainrot
                row["judge_model"] = args.model_id
                row["judge_raw_response"] = raw_response
                writer.writerow(row)

    print(f"Done! Judged results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

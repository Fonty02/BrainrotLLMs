"""
Generate HTCondor submit files for steering_experiment.py jobs.

Each job handles one (model, technique, layer_pct) combination.
Optionally splits questions across --q_slice workers.
"""

import itertools
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config_steering.yaml"


def model_short(model_id: str) -> str:
    mapping = {
        "Qwen/Qwen2.5-7B-Instruct": "Qwen7B",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama8B",
        "google/gemma-4-E2B-it": "Gemma2B",
    }
    return mapping.get(model_id, model_id.replace("/", "_"))


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    htc = cfg["htc"]

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    results_dir = PROJECT_ROOT / htc["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = common["models"]
    techniques = common["techniques"]
    layer_pcts = common["layer_pcts"]
    q_slices = int(common.get("q_slices", 1))

    submit_lines = []
    job_count = 0

    for model in models:
        for technique in techniques:
            for layer_pct in layer_pcts:
                short = model_short(model)
                base_name = f"{short}_{technique}_L{layer_pct}"

                for slice_idx in range(1, q_slices + 1):
                    # Output CSV: each job writes its own file
                    if q_slices > 1:
                        csv_name = f"{base_name}_q{slice_idx}of{q_slices}.csv"
                        q_slice_arg = f"{slice_idx}/{q_slices}"
                    else:
                        csv_name = f"{base_name}.csv"
                        q_slice_arg = ""

                    csv_path = results_dir / csv_name

                    args = [
                        "--model", model,
                        "--technique", technique,
                        "--layer_pct", str(layer_pct),
                        "--output_csv", str(csv_path),
                        "--device", common.get("device", "cuda"),
                        "--max_pairs", str(common.get("max_pairs", 2000)),
                        "--layer_band", str(common.get("layer_band", 0)),
                        "--repetition_penalty", str(common.get("repetition_penalty", 1.3)),
                        "--max_new_tokens", str(common.get("max_new_tokens", 200)),
                    ]

                    # Per-technique coefficients (fall back to legacy single key,
                    # then to the script's per-technique defaults).
                    coeffs_by_tech = common.get("coefficients_by_technique", {}) or {}
                    coeff_val = coeffs_by_tech.get(technique, common.get("coefficients"))
                    if coeff_val:
                        args.extend(["--coefficients", coeff_val])

                    if common.get("do_sample"):
                        args.extend([
                            "--do_sample",
                            "--temperature", str(common.get("temperature", 0.8)),
                            "--top_p", str(common.get("top_p", 0.9)),
                        ])
                    if common.get("hf_cache_dir"):
                        args.extend(["--hf_cache_dir", common["hf_cache_dir"]])
                    if common.get("hf_token"):
                        args.extend(["--hf_token", common["hf_token"]])
                    if q_slice_arg:
                        args.extend(["--q_slice", q_slice_arg])

                    submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')
                    job_count += 1

    print(f"Total jobs to generate: {job_count}")
    print(f"  - Models: {len(models)} ({', '.join(model_short(m) for m in models)})")
    print(f"  - Techniques: {len(techniques)} ({', '.join(techniques)})")
    print(f"  - Layers: {len(layer_pcts)} ({', '.join(str(l) for l in layer_pcts)})")
    print(f"  - Q-slices per config: {q_slices}")

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 27))
    executable = htc["executable"]

    htc_header_lines = [
        "universe = vanilla",
        f"executable = {executable}",
        f"request_cpus = {htc['request_cpus']}",
        f"request_gpus = {htc['request_gpus']}",
    ]

    if "request_memory" in htc:
        htc_header_lines.append(f"request_memory = {htc['request_memory']}")
    if "request_disk" in htc:
        htc_header_lines.append(f"request_disk = {htc['request_disk']}")
    if "initialdir" in htc:
        htc_header_lines.append(f"initialdir = {htc['initialdir']}")
    if "getenv" in htc:
        htc_header_lines.append(f"getenv = {htc['getenv']}")
    if "requirements" in htc:
        htc_header_lines.append(f"requirements = {htc['requirements']}")

    htc_header_lines.extend([
        f"log = {htc['logs_dir']}/job_$(Cluster)_$(Process).log",
        f"output = {htc['logs_dir']}/job_$(Cluster)_$(Process).out",
        f"error = {htc['logs_dir']}/job_$(Cluster)_$(Process).err",
        "",
    ])

    # Remove stale files
    for old_file in output_dir.glob("steering_jobs_*.htc"):
        old_file.unlink()

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx:idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"steering_jobs_{file_num}.htc"

        with open(submit_path, "w", encoding="utf-8") as f:
            f.write("\n".join(htc_header_lines))
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    # Generate convenience script to submit all files
    submit_all_path = output_dir / "submit_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated HTCondor steering job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
        for i in range(1, num_files + 1):
            f.write(f'condor_submit "$SCRIPT_DIR/steering_jobs_{i}.htc"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all jobs")


if __name__ == "__main__":
    main()

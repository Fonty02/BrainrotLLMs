"""
Generate HTCondor submit files for judge.py jobs.

Each job processes one steering CSV (or a slice of it via --row_slice).
Input CSVs can be specified explicitly or auto-discovered from a directory.
"""

import glob
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config_judge.yaml"


def _resolve_inputs(cfg_common: dict, project_root: Path) -> list[Path]:
    """Resolve input CSV list from explicit paths or directory glob."""
    csvs = []

    if cfg_common.get("input_csvs"):
        for entry in cfg_common["input_csvs"]:
            p = Path(entry)
            if not p.is_absolute():
                p = project_root / p
            if "*" in str(p) or "?" in str(p):
                csvs.extend(Path(f) for f in sorted(glob.glob(str(p), recursive=True)))
            else:
                csvs.append(p)
    elif cfg_common.get("input_dir"):
        input_dir = project_root / cfg_common["input_dir"]
        if input_dir.is_dir():
            csvs = sorted(input_dir.glob("*.csv"))
            print(f"Auto-discovered {len(csvs)} CSV(s) in {input_dir}")

    csvs = [c for c in csvs if not c.name.endswith("_judged.csv")]
    return [c for c in csvs if c.exists()]


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    htc = cfg["htc"]

    output_dir = PROJECT_ROOT / htc["output_dir"]
    logs_dir = PROJECT_ROOT / htc["logs_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_csvs = _resolve_inputs(common, PROJECT_ROOT)
    if not input_csvs:
        print("ERROR: No input CSVs found. Check config_judge.yaml input_csvs or input_dir.")
        return

    row_slices = int(common.get("row_slices", 1))

    submit_lines = []
    job_count = 0

    for csv_path in input_csvs:
        stem = csv_path.stem

        for slice_idx in range(1, row_slices + 1):
            if row_slices > 1:
                output_csv = csv_path.parent / f"{stem}_judged_s{slice_idx}of{row_slices}.csv"
                row_slice_arg = f"{slice_idx}/{row_slices}"
            else:
                output_csv = csv_path.parent / f"{stem}_judged.csv"
                row_slice_arg = ""

            args = [
                "--input_csv", str(csv_path),
                "--output_csv", str(output_csv),
                "--model_id", common.get("judge_model", "google/gemma-4-E4B-it"),
                "--batch_size", str(common.get("batch_size", 16)),
            ]
            if common.get("hf_cache_dir"):
                args.extend(["--hf_cache_dir", common["hf_cache_dir"]])
            if row_slice_arg:
                args.extend(["--row_slice", row_slice_arg])

            flat = " ".join(args)
            submit_lines.append(f'arguments = "{flat}"\nqueue\n')
            job_count += 1

    print(f"Total judge jobs: {job_count}")
    print(f"  - Input CSVs: {len(input_csvs)}")
    print(f"  - Row slices per CSV: {row_slices}")

    max_jobs_per_file = int(htc.get("max_jobs_per_file", 30))
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
        f"log = {htc['logs_dir']}/judge_$(Cluster)_$(Process).log",
        f"output = {htc['logs_dir']}/judge_$(Cluster)_$(Process).out",
        f"error = {htc['logs_dir']}/judge_$(Cluster)_$(Process).err",
        "",
    ])

    # Remove stale files
    for old_file in output_dir.glob("judge_jobs_*.htc"):
        old_file.unlink()

    for idx in range(0, len(submit_lines), max_jobs_per_file):
        subset = submit_lines[idx:idx + max_jobs_per_file]
        file_num = idx // max_jobs_per_file + 1
        submit_path = output_dir / f"judge_jobs_{file_num}.htc"

        with open(submit_path, "w", encoding="utf-8") as f:
            f.write("\n".join(htc_header_lines))
            f.writelines(subset)

        print(f"Created submit file: {submit_path} with {len(subset)} jobs")

    submit_all_path = output_dir / "submit_judge_all.sh"
    with open(submit_all_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all generated HTCondor judge job files\n\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n\n')
        num_files = (len(submit_lines) + max_jobs_per_file - 1) // max_jobs_per_file
        for i in range(1, num_files + 1):
            f.write(f'condor_submit "$SCRIPT_DIR/judge_jobs_{i}.htc"\n')

    print(f"\nCreated submit helper: {submit_all_path}")
    print(f"Run 'bash {submit_all_path.relative_to(PROJECT_ROOT)}' to submit all judge jobs")


if __name__ == "__main__":
    main()

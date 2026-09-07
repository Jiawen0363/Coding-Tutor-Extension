#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


DEFAULT_EVO_ROOT = Path("/data/wangjian/EvoCodeBench/EvoCodeBench-2604")
DEFAULT_LEVELS = ("low_level", "med_level", "high_level")


OOM_PATTERNS = (
    "out of memory",
    "out_of_memory",
    "cuda error: out of memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Remove OOM entries from post-test pass logs and rerun only those "
            "completions with the existing pass_k.py evaluator."
        )
    )
    parser.add_argument(
        "posttest_dir",
        type=Path,
        help=(
            "Directory containing level subdirectories, e.g. "
            "output/student_posttest/vanilla/backbone/gpt-4o"
        ),
    )
    parser.add_argument(
        "--source_code_root",
        type=Path,
        default=DEFAULT_EVO_ROOT / "Source_Code",
        help="EvoCodeBench Source_Code directory.",
    )
    parser.add_argument(
        "--data_file",
        type=Path,
        default=DEFAULT_EVO_ROOT / "metadata.jsonl",
        help="EvoCodeBench metadata.jsonl file.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Student level directories to scan.",
    )
    parser.add_argument("--k", default="1,3,5,10", help="Pass@k values for pass_k.py.")
    parser.add_argument("--n", type=int, default=10, help="Number of completions per task.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run pass_k.py and check_source_code.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report OOM counts without modifying or rerunning anything.",
    )
    parser.add_argument(
        "--skip-check-source-code",
        action="store_true",
        help="Skip traver/utils/check_source_code.py before rerunning each level.",
    )
    return parser.parse_args()


def error_log_text(record):
    error_log = record.get("error_log")
    if not isinstance(error_log, dict):
        return ""
    fields = [
        error_log.get("reason", ""),
        error_log.get("stdout", ""),
        error_log.get("stderr", ""),
        error_log.get("exception", ""),
        error_log.get("test", ""),
    ]
    return "\n".join(str(field) for field in fields if field is not None).lower()


def is_oom_record(record):
    if record.get("Result") != "Fail":
        return False
    text = error_log_text(record)
    return any(pattern in text for pattern in OOM_PATTERNS)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def repo_root():
    return Path(__file__).resolve().parents[2]


def run_command(command, cwd):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def rerun_level(args, root, level, timestamp):
    level_dir = args.posttest_dir / level
    completion_file = level_dir / "completion.jsonl"
    log_file = level_dir / "test_results.jsonl"

    if not log_file.exists():
        print(f"[skip] {level}: missing {log_file}")
        return 0
    if not completion_file.exists():
        print(f"[skip] {level}: missing {completion_file}")
        return 0

    records = read_jsonl(log_file)
    oom_records = [record for record in records if is_oom_record(record)]
    if not oom_records:
        print(f"[skip] {level}: no OOM records")
        return 0

    print(f"[found] {level}: {len(oom_records)} OOM records")
    if args.dry_run:
        return len(oom_records)

    backup_file = log_file.with_name(f"{log_file.name}.bak.{timestamp}")
    shutil.copy2(log_file, backup_file)

    non_oom_records = [record for record in records if not is_oom_record(record)]
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{log_file.stem}.rerun_oom.",
        suffix=".jsonl",
        dir=level_dir,
        text=True,
    )
    os.close(fd)
    temp_log_file = Path(temp_name)
    write_jsonl(temp_log_file, non_oom_records)

    try:
        if not args.skip_check_source_code:
            run_command(
                [
                    args.python,
                    "traver/utils/check_source_code.py",
                    str(args.source_code_root),
                ],
                cwd=root,
            )

        run_command(
            [
                args.python,
                "traver/parser/pass_k.py",
                "--output_file",
                str(completion_file),
                "--log_file",
                str(temp_log_file),
                "--data_file",
                str(args.data_file),
                "--source_code_root",
                str(args.source_code_root),
                "--k",
                args.k,
                "--n",
                str(args.n),
            ],
            cwd=root,
        )
    except Exception:
        print(f"[error] {level}: original log kept at {log_file}")
        print(f"[error] {level}: partial rerun log kept at {temp_log_file}")
        raise

    os.replace(temp_log_file, log_file)
    print(f"[done] {level}: replaced {log_file}")
    print(f"[backup] {level}: {backup_file}")
    return len(oom_records)


def main():
    args = parse_args()
    args.posttest_dir = args.posttest_dir.resolve()
    root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.posttest_dir.exists():
        raise FileNotFoundError(f"posttest_dir does not exist: {args.posttest_dir}")
    if not args.dry_run and not args.data_file.exists():
        raise FileNotFoundError(f"data_file does not exist: {args.data_file}")
    if not args.dry_run and not args.source_code_root.exists():
        raise FileNotFoundError(f"source_code_root does not exist: {args.source_code_root}")

    print(f"Post-test dir: {args.posttest_dir}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    total = 0
    for level in args.levels:
        total += rerun_level(args, root, level, timestamp)

    if args.dry_run:
        print(f"Dry run complete. Total OOM records: {total}")
    else:
        print(f"Rerun complete. Total OOM records requested: {total}")


if __name__ == "__main__":
    main()

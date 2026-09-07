#!/usr/bin/env python3
"""Filter dialogue data by namespace-level post-test results.

The script matches each dialogue level independently. For example,
``high_level/simulated_dialogs.json`` is filtered only with namespaces from
``high_level/test_results.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DIALOGUE_DIR = Path("output/dialogue/vanilla/gpt-4o")
DEFAULT_LABEL_DIR = Path("output/student_posttest/vanilla/backbone/gpt-4o")
DEFAULT_OUTPUT_DIR = Path("output/dialogue/vanilla/gpt-4o_failed")
DEFAULT_LEVELS = ("low_level", "med_level", "high_level")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter gpt-4o dialogue files level-by-level, keeping only "
            "namespaces that match the requested post-test result policy."
        )
    )
    parser.add_argument("--dialogue-dir", type=Path, default=DEFAULT_DIALOGUE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument(
        "--label-file",
        default="test_results.jsonl",
        help="JSONL file under each level directory that contains namespace and Result.",
    )
    parser.add_argument(
        "--dialogue-file",
        default="simulated_dialogs.json",
        help="JSON file under each level directory that contains namespace and conversation.",
    )
    parser.add_argument(
        "--fail-policy",
        choices=("any", "all"),
        default="all",
        help=(
            "'any' keeps namespaces with at least one target-result row; "
            "'all' keeps only namespaces whose rows are all target-result. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--target-result",
        default="Fail",
        help="Result value to select, for example Fail or Pass. Default: Fail.",
    )
    parser.add_argument(
        "--limit-per-level",
        nargs="*",
        default=[],
        metavar="LEVEL=N",
        help="Optionally keep only the first N filtered dialogues for each level.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_limit_per_level(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --limit-per-level value: {value}. Expected LEVEL=N.")
        level, limit_text = value.split("=", 1)
        level = level.strip()
        if not level:
            raise ValueError(f"Invalid empty level in --limit-per-level value: {value}")
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise ValueError(f"Invalid limit in --limit-per-level value: {value}") from exc
        if limit < 0:
            raise ValueError(f"Limit must be non-negative in --limit-per-level value: {value}")
        limits[level] = limit
    return limits


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def collect_failed_namespaces(
    label_path: Path, fail_policy: str, target_result: str
) -> tuple[set[str], set[str], dict[str, Any]]:
    rows = iter_jsonl(label_path)
    result_by_namespace: dict[str, list[str]] = defaultdict(list)
    missing_namespace_rows = 0

    for row in rows:
        namespace = row.get("namespace")
        result = row.get("Result")
        if not namespace:
            missing_namespace_rows += 1
            continue
        result_by_namespace[str(namespace)].append(str(result).strip())

    selected_namespaces: set[str] = set()
    mixed_namespaces: list[str] = []
    target_result_normalized = target_result.casefold()
    for namespace, results in result_by_namespace.items():
        result_values = {result.casefold() for result in results}
        has_fail = "fail" in result_values
        has_pass = "pass" in result_values
        has_target = target_result_normalized in result_values
        has_non_target = any(result != target_result_normalized for result in result_values)

        if has_fail and has_pass:
            mixed_namespaces.append(namespace)

        if fail_policy == "any" and has_target:
            selected_namespaces.add(namespace)
        elif fail_policy == "all" and results and not has_non_target:
            selected_namespaces.add(namespace)

    stats = {
        "label_rows": len(rows),
        "label_namespaces": len(result_by_namespace),
        "result_counts": dict(Counter(result for results in result_by_namespace.values() for result in results)),
        "missing_namespace_rows": missing_namespace_rows,
        "mixed_pass_fail_namespaces": sorted(mixed_namespaces),
        "target_result": target_result,
        "selected_namespaces": len(selected_namespaces),
    }
    return selected_namespaces, set(result_by_namespace), stats


def filter_dialogues(
    dialogue_path: Path, selected_namespaces: set[str], label_namespaces: set[str]
) -> tuple[list[Any], dict[str, Any]]:
    dialogues = read_json(dialogue_path)
    if not isinstance(dialogues, list):
        raise ValueError(f"Expected a JSON list in {dialogue_path}")

    filtered: list[Any] = []
    missing_namespace_items = 0
    duplicate_counts: Counter[str] = Counter()

    for item in dialogues:
        if not isinstance(item, dict):
            raise ValueError(f"Expected every dialogue item in {dialogue_path} to be an object")
        namespace = item.get("namespace")
        if not namespace:
            missing_namespace_items += 1
            continue
        namespace = str(namespace)
        duplicate_counts[namespace] += 1
        if namespace in selected_namespaces:
            filtered.append(item)

    dialogue_namespaces = set(duplicate_counts)
    stats = {
        "dialogue_items": len(dialogues),
        "dialogue_namespaces": len(dialogue_namespaces),
        "missing_namespace_items": missing_namespace_items,
        "duplicate_dialogue_namespaces": {
            namespace: count for namespace, count in sorted(duplicate_counts.items()) if count > 1
        },
        "filtered_dialogue_items": len(filtered),
        "filtered_dialogue_namespaces": len({item["namespace"] for item in filtered}),
        "selected_namespaces_without_dialogue": sorted(selected_namespaces - dialogue_namespaces),
        "dialogue_namespaces_without_label": sorted(dialogue_namespaces - label_namespaces),
        "dialogue_namespaces_not_selected": sorted(dialogue_namespaces - selected_namespaces),
    }
    return filtered, stats


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists, use --overwrite to replace it: {path}")


def main() -> None:
    args = parse_args()
    limits = parse_limit_per_level(args.limit_per_level)
    summary: dict[str, Any] = {
        "dialogue_dir": str(args.dialogue_dir),
        "label_dir": str(args.label_dir),
        "output_dir": str(args.output_dir),
        "label_file": args.label_file,
        "dialogue_file": args.dialogue_file,
        "fail_policy": args.fail_policy,
        "target_result": args.target_result,
        "limit_per_level": limits,
        "levels": {},
    }

    for level in args.levels:
        dialogue_path = args.dialogue_dir / level / args.dialogue_file
        label_path = args.label_dir / level / args.label_file
        output_path = args.output_dir / level / args.dialogue_file

        if not dialogue_path.exists():
            raise FileNotFoundError(f"Missing dialogue file for {level}: {dialogue_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {level}: {label_path}")
        ensure_output_path(output_path, args.overwrite)

        selected_namespaces, label_namespaces, label_stats = collect_failed_namespaces(
            label_path, args.fail_policy, args.target_result
        )
        filtered_dialogues, dialogue_stats = filter_dialogues(
            dialogue_path, selected_namespaces, label_namespaces
        )
        filtered_before_limit = len(filtered_dialogues)
        if level in limits:
            limit = limits[level]
            if filtered_before_limit < limit:
                raise ValueError(
                    f"Cannot keep {limit} dialogues for {level}; only "
                    f"{filtered_before_limit} dialogues matched the filter."
                )
            filtered_dialogues = filtered_dialogues[:limit]
            dialogue_stats["filtered_dialogue_items_before_limit"] = filtered_before_limit
            dialogue_stats["filtered_dialogue_namespaces_before_limit"] = dialogue_stats[
                "filtered_dialogue_namespaces"
            ]
            dialogue_stats["limit"] = limit
            dialogue_stats["filtered_dialogue_items"] = len(filtered_dialogues)
            dialogue_stats["filtered_dialogue_namespaces"] = len(
                {item["namespace"] for item in filtered_dialogues}
            )
        write_json(output_path, filtered_dialogues)

        level_summary = {
            **label_stats,
            **dialogue_stats,
            "output_file": str(output_path),
        }
        summary["levels"][level] = level_summary

        print(
            f"{level}: kept {dialogue_stats['filtered_dialogue_items']} / "
            f"{dialogue_stats['dialogue_items']} dialogues "
            f"({label_stats['selected_namespaces']} namespaces with "
            f"{args.target_result} by policy={args.fail_policy})"
        )

    summary_path = args.output_dir / "filter_summary.json"
    ensure_output_path(summary_path, args.overwrite)
    write_json(summary_path, summary)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

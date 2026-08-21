#!/usr/bin/env python3
"""Aggregate target-API zero-shot, fine-tuned, and learning-curve runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run root to scan. Repeat to combine seeds across run roots.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_result_path(run_dir: Path, path: Path):
    relative = path.relative_to(run_dir)
    parts = relative.parts
    if len(parts) < 4 or not parts[1].startswith("seed-"):
        return None
    task = parts[0]
    seed = int(parts[1].removeprefix("seed-"))
    if "target-zero-shot-summary" in parts:
        return task, seed, "zero_shot", None
    if path.parent.name != "oof-summary":
        return None
    size_match = re.fullmatch(r"size-(\d+)", path.parent.parent.name)
    if size_match:
        return task, seed, "fine_tuned_learning_curve", int(
            size_match.group(1)
        )
    return task, seed, "fine_tuned_full", None


def _fine_tuning_pairs(path: Path, stage: str, size: int | None) -> int:
    if stage == "zero_shot":
        return 0
    if size is not None:
        return size
    seed_dir = path.parent.parent
    config_path = seed_dir / "fold-0" / "final" / "run_config.json"
    if not config_path.is_file():
        return 0
    config = _read_json(config_path)
    explicit = config.get("fine_tuning_physical_pairs")
    if explicit is not None:
        return int(explicit)
    split_manifest = seed_dir / "fold-0" / "selection" / "split_manifest.csv"
    if split_manifest.is_file():
        with split_manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "pair_key" in rows[0]:
            return len({row["pair_key"] for row in rows})
    train_rows = int(config.get("train_rows", 0))
    return train_rows


def _flatten_metrics(summary: dict) -> dict[str, float | int]:
    pooled = summary["pooled"]
    metrics: dict[str, float | int] = {
        "pairs": int(pooled["pairs"]),
        "accuracy": float(pooled["accuracy"]),
        "balanced_accuracy": float(
            pooled["balanced_accuracy_present_classes"]
        ),
    }
    for class_name, values in sorted(pooled["per_class"].items()):
        if values["accuracy"] is not None:
            metrics[f"class_{class_name}_recall"] = float(values["accuracy"])
        metrics[f"class_{class_name}_f1"] = float(values["f1"])
        metrics[f"class_{class_name}_support"] = int(values["support"])
    for api_name, values in sorted(summary["by_target_api"].items()):
        key = api_name.lower().replace(" ", "_")
        metrics[f"api_{key}_accuracy"] = float(values["accuracy"])
        metrics[f"api_{key}_balanced_accuracy"] = float(
            values["balanced_accuracy_present_classes"]
        )
    return metrics


def collect_results(run_dirs: list[str | Path]) -> list[dict]:
    rows = []
    seen = set()
    for run_value in run_dirs:
        run_dir = Path(run_value)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        for path in sorted(run_dir.glob("**/summary_metrics.json")):
            parsed = _parse_result_path(run_dir, path)
            if parsed is None:
                continue
            task, seed, stage, size = parsed
            key = (task, seed, stage, size)
            if key in seen:
                raise ValueError(f"Duplicate result for {key}")
            seen.add(key)
            summary = _read_json(path)
            if summary.get("task") != task:
                raise ValueError(f"Task mismatch in {path}")
            row = {
                "task": task,
                "stage": stage,
                "fine_tuning_pairs": _fine_tuning_pairs(path, stage, size),
                "seed": seed,
                "source_run": str(run_dir),
                "summary_path": str(path),
            }
            row.update(_flatten_metrics(summary))
            rows.append(row)
    if not rows:
        raise ValueError("No target-API summary results found")
    return sorted(
        rows,
        key=lambda row: (
            row["task"],
            row["stage"],
            row["fine_tuning_pairs"],
            row["seed"],
        ),
    )


def aggregate_results(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[
            (row["task"], row["stage"], row["fine_tuning_pairs"])
        ].append(row)

    aggregate = []
    metadata = {
        "task",
        "stage",
        "fine_tuning_pairs",
        "seed",
        "source_run",
        "summary_path",
    }
    for (task, stage, fine_tuning_pairs), group in sorted(groups.items()):
        numeric_keys = sorted(
            set.intersection(
                *(set(row).difference(metadata) for row in group)
            )
        )
        result = {
            "task": task,
            "stage": stage,
            "fine_tuning_pairs": fine_tuning_pairs,
            "runs": len(group),
            "seeds": ",".join(
                str(row["seed"])
                for row in sorted(group, key=lambda item: item["seed"])
            ),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate.append(result)
    return aggregate


def _write_csv(path: Path, rows: list[dict]):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _display_stage(stage: str) -> str:
    return {
        "zero_shot": "Zero-shot pretraining model",
        "fine_tuned_full": "Five-fold fine-tuning",
        "fine_tuned_learning_curve": "Fine-tuning learning curve",
    }[stage]


def _mean_std(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def _write_markdown(path: Path, aggregate: list[dict]):
    lines = [
        "# Three-API experiment summary",
        "",
        "| Task | Evaluation | Fine-tuning pairs | Seeds | Accuracy | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    _display_stage(row["stage"]),
                    str(row["fine_tuning_pairs"]),
                    str(row["runs"]),
                    _mean_std(row, "accuracy"),
                    _mean_std(row, "balanced_accuracy"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(output_dir: str | Path, rows: list[dict]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_results(rows)
    raw_path = output_dir / "target_api_runs.csv"
    aggregate_path = output_dir / "target_api_aggregate.csv"
    json_path = output_dir / "target_api_summary.json"
    markdown_path = output_dir / "target_api_summary.md"
    _write_csv(raw_path, rows)
    _write_csv(aggregate_path, aggregate)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "mcc-gcn-target-api-multiseed-summary-v1",
                "runs": rows,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown_path, aggregate)
    return raw_path, aggregate_path, json_path, markdown_path


def main():
    args = parse_args()
    rows = collect_results(args.run_dir)
    outputs = write_summary(args.output_dir, rows)
    print(f"Summarized {len(rows)} target-API results")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

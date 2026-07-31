#!/usr/bin/env python3
"""Summarize the locked pretraining class-imbalance ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = {
    "binary": ["negative", "positive"],
    "four-class": [
        "negative",
        "salt",
        "cocrystal",
        "hydrate_or_solvate",
    ],
}
SUMMARY_METRICS = [
    "validation_balanced_accuracy",
    "validation_loss",
    "external_overall_accuracy",
    "external_balanced_accuracy",
]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_paths(run_dir, task, weighting, seed):
    pretrain = run_dir / task / weighting / f"seed-{seed}" / "pretrain"
    return {
        "config": pretrain / "run_config.json",
        "selection": pretrain / "selection_result.json",
        "metrics": pretrain / "external_64_predictions.metrics.json",
    }


def _counts_from_confusion_matrix(matrix, class_names):
    if len(matrix) != len(class_names) or any(
        len(row) != len(class_names) for row in matrix
    ):
        raise ValueError("Confusion matrix does not match configured task")
    true_counts = [sum(row) for row in matrix]
    predicted_counts = [
        sum(row[index] for row in matrix)
        for index in range(len(class_names))
    ]
    return true_counts, predicted_counts


def collect_results(run_dir, *, allow_incomplete=False):
    run_dir = Path(run_dir)
    profile_path = run_dir / "experiment_profile.json"
    profile = _read_json(profile_path)
    expected = [
        (task, weighting, int(seed))
        for task in profile["tasks"]
        for weighting in profile["class_weightings"]
        for seed in profile["seeds"]
    ]
    rows = []
    missing = []
    for task, weighting, seed in expected:
        paths = _result_paths(run_dir, task, weighting, seed)
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue

        config = _read_json(paths["config"])
        selection = _read_json(paths["selection"])
        metrics = _read_json(paths["metrics"])
        if config["task"] != task or metrics["task"] != task:
            raise ValueError(f"Task metadata mismatch for {paths['config']}")
        if config["class_weighting"] != weighting:
            raise ValueError(
                f"Class-weighting metadata mismatch for {paths['config']}"
            )
        if int(config["seed"]) != seed:
            raise ValueError(f"Seed metadata mismatch for {paths['config']}")

        class_names = CLASS_NAMES[task]
        matrix = metrics["confusion_matrix"]
        true_counts, predicted_counts = _counts_from_confusion_matrix(
            matrix,
            class_names,
        )
        row = {
            "task": task,
            "class_weighting": weighting,
            "seed": seed,
            "validation_aggregation": config["validation_aggregation"],
            "best_epoch": int(selection["best_epoch"]),
            "epochs_completed": int(selection["epochs_completed"]),
            "validation_balanced_accuracy": float(
                selection["best_validation_balanced_accuracy"]
            ),
            "validation_loss": float(selection["best_validation_loss"]),
            "external_overall_accuracy": float(metrics["overall_accuracy"]),
            "external_balanced_accuracy": float(metrics["balanced_accuracy"]),
        }
        for index, class_name in enumerate(class_names):
            row[f"external_{class_name}_accuracy"] = float(
                metrics["per_class_accuracy"][class_name]
            )
            row[f"external_{class_name}_true_count"] = true_counts[index]
            row[f"external_{class_name}_predicted_count"] = (
                predicted_counts[index]
            )
        rows.append(row)

    if missing and not allow_incomplete:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Incomplete ablation results:\n{formatted}")
    return profile, rows, missing


def aggregate_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["class_weighting"])].append(row)

    aggregated = []
    for (task, weighting), group in sorted(grouped.items()):
        metric_names = SUMMARY_METRICS + [
            f"external_{class_name}_accuracy"
            for class_name in CLASS_NAMES[task]
        ]
        result = {
            "task": task,
            "class_weighting": weighting,
            "runs": len(group),
            "seeds": ",".join(
                str(row["seed"])
                for row in sorted(group, key=lambda item: item["seed"])
            ),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group]
            result[f"{metric_name}_mean"] = statistics.fmean(values)
            result[f"{metric_name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregated.append(result)
    return aggregated


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(run_dir, profile, rows, missing):
    run_dir = Path(run_dir)
    aggregated = aggregate_results(rows)
    raw_csv = run_dir / "pretrain_imbalance_runs.csv"
    aggregate_csv = run_dir / "pretrain_imbalance_summary.csv"
    summary_json = run_dir / "pretrain_imbalance_summary.json"
    _write_csv(raw_csv, rows)
    _write_csv(aggregate_csv, aggregated)
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "mcc-gcn-pretrain-imbalance-summary-v1",
                "profile": profile,
                "runs": rows,
                "aggregated": aggregated,
                "missing_files": missing,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return raw_csv, aggregate_csv, summary_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial summary instead of failing on missing runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    profile, rows, missing = collect_results(
        args.run_dir,
        allow_incomplete=args.allow_incomplete,
    )
    outputs = write_summary(args.run_dir, profile, rows, missing)
    print(f"Summarized {len(rows)} runs")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

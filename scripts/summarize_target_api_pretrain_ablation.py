#!/usr/bin/env python3
"""Aggregate the target-API pretraining undersampling comparison."""

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


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_paths(run_dir, task, strategy, seed):
    pretrain = run_dir / task / strategy / f"seed-{seed}" / "pretrain"
    return {
        "config": pretrain / "run_config.json",
        "selection": pretrain / "selection_result.json",
        "evaluation": pretrain / "target_predictions.metrics.json",
        "target_summary": pretrain / "target-summary" / "summary_metrics.json",
    }


def _predicted_counts(matrix):
    return [sum(row[index] for row in matrix) for index in range(len(matrix))]


def collect_results(run_dir, *, allow_incomplete=False):
    run_dir = Path(run_dir)
    profile = _read_json(run_dir / "experiment_profile.json")
    expected = [
        (task, strategy, int(seed))
        for task in profile["tasks"]
        for strategy in profile["enabled_strategies"]
        for seed in profile["seeds"]
    ]
    rows = []
    missing = []
    for task, strategy, seed in expected:
        paths = _result_paths(run_dir, task, strategy, seed)
        absent = [str(path) for path in paths.values() if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue

        strategy_profile = profile["strategies"][strategy]
        feature_profile = strategy_profile["features_by_task"][task]
        config = _read_json(paths["config"])
        selection = _read_json(paths["selection"])
        evaluation = _read_json(paths["evaluation"])
        summary = _read_json(paths["target_summary"])
        if config["task"] != task or evaluation["task"] != task:
            raise ValueError(f"Task metadata mismatch for {paths['config']}")
        if int(config["seed"]) != seed:
            raise ValueError(f"Seed metadata mismatch for {paths['config']}")
        if config["class_weighting"] != strategy_profile["class_weighting"]:
            raise ValueError(f"Class-weighting metadata mismatch for {paths['config']}")
        if (
            Path(config["resolved_npz"]).resolve()
            != Path(feature_profile["path"]).resolve()
        ):
            raise ValueError(f"Training feature mismatch for {paths['config']}")

        pooled = summary["pooled"]
        matrix = pooled["confusion_matrix"]
        class_names = CLASS_NAMES[task]
        if len(matrix) != len(class_names):
            raise ValueError(f"Confusion matrix mismatch in {paths['target_summary']}")
        predicted_counts = _predicted_counts(matrix)
        per_class = pooled["per_class"]
        row = {
            "task": task,
            "strategy": strategy,
            "seed": seed,
            "class_weighting": config["class_weighting"],
            "training_physical_pairs": int(feature_profile["physical_pairs"]),
            "training_ordered_rows": int(feature_profile["ordered_rows"]),
            "best_epoch": int(selection["best_epoch"]),
            "epochs_completed": int(selection["epochs_completed"]),
            "source_validation_balanced_accuracy": float(
                selection["best_validation_balanced_accuracy"]
            ),
            "source_validation_loss": float(selection["best_validation_loss"]),
            "target_accuracy": float(pooled["accuracy"]),
            "target_balanced_accuracy": float(
                pooled["balanced_accuracy_present_classes"]
            ),
            "target_macro_f1": statistics.fmean(
                float(per_class[name]["f1"]) for name in class_names
            ),
            "target_negative_false_positive_rate": 1.0
            - float(per_class["negative"]["recall"]),
        }
        for index, name in enumerate(class_names):
            values = per_class[name]
            row[f"target_{name}_recall"] = float(values["recall"])
            row[f"target_{name}_precision"] = float(values["precision"])
            row[f"target_{name}_f1"] = float(values["f1"])
            row[f"target_{name}_support"] = int(values["support"])
            row[f"target_{name}_predicted_count"] = int(predicted_counts[index])
        for api_name, values in sorted(summary["by_target_api"].items()):
            key = api_name.lower().replace(" ", "_")
            row[f"api_{key}_accuracy"] = float(values["accuracy"])
            row[f"api_{key}_balanced_accuracy"] = float(
                values["balanced_accuracy_present_classes"]
            )
        rows.append(row)

    if missing and not allow_incomplete:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Incomplete ablation results:\n{formatted}")
    return profile, rows, missing


def aggregate_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["strategy"])].append(row)

    metadata = {
        "task",
        "strategy",
        "seed",
        "class_weighting",
        "training_physical_pairs",
        "training_ordered_rows",
    }
    aggregated = []
    for (task, strategy), group in sorted(grouped.items()):
        numeric_keys = sorted(set.intersection(*(set(row) for row in group)) - metadata)
        result = {
            "task": task,
            "strategy": strategy,
            "training_physical_pairs": group[0]["training_physical_pairs"],
            "training_ordered_rows": group[0]["training_ordered_rows"],
            "runs": len(group),
            "seeds": ",".join(
                str(row["seed"])
                for row in sorted(group, key=lambda value: value["seed"])
            ),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
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


def _mean_std(row, metric):
    return f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"


def _write_markdown(path, aggregated):
    lines = [
        "# Target-API pretraining undersampling ablation",
        "",
        "| Task | Training strategy | Physical pairs | Seeds | Source validation BAcc | Target accuracy | Target BAcc | Target macro F1 | Negative FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["strategy"],
                    str(row["training_physical_pairs"]),
                    str(row["runs"]),
                    _mean_std(row, "source_validation_balanced_accuracy"),
                    _mean_std(row, "target_accuracy"),
                    _mean_std(row, "target_balanced_accuracy"),
                    _mean_std(row, "target_macro_f1"),
                    _mean_std(row, "target_negative_false_positive_rate"),
                ]
            )
            + " |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(run_dir, profile, rows, missing):
    run_dir = Path(run_dir)
    aggregated = aggregate_results(rows)
    raw_csv = run_dir / "pretrain_ablation_runs.csv"
    aggregate_csv = run_dir / "pretrain_ablation_summary.csv"
    summary_json = run_dir / "pretrain_ablation_summary.json"
    markdown = run_dir / "pretrain_ablation_summary.md"
    _write_csv(raw_csv, rows)
    _write_csv(aggregate_csv, aggregated)
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "mcc-gcn-target-pretrain-undersampling-summary-v1",
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
    _write_markdown(markdown, aggregated)
    return raw_csv, aggregate_csv, summary_json, markdown


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    profile, rows, missing = collect_results(
        args.run_dir,
        allow_incomplete=args.allow_incomplete,
    )
    outputs = write_summary(args.run_dir, profile, rows, missing)
    print(f"Summarized {len(rows)} ablation runs")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

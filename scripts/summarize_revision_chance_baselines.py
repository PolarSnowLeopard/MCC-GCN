#!/usr/bin/env python3
"""Calculate distribution-aware chance baselines for revision datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcc_gcn.models.metrics import calculate_chance_baselines

FOUR_CLASS_NAMES = ["negative", "salt", "cocrystal", "hydrate_or_solvate"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-manifest",
        default="data/manifests/external-64-split-v1.csv",
    )
    parser.add_argument(
        "--target-table",
        default=("runs/revision-three-api/data/four_class_target_physical_pairs.csv"),
    )
    parser.add_argument(
        "--source-validation-table",
        default=(
            "runs/revision-three-api/data/pretrain-split/validation_physical_pairs.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _labels(table, description):
    if "label_int" not in table:
        raise ValueError(f"{description} is missing label_int")
    labels = pd.to_numeric(table["label_int"], errors="raise").to_numpy(dtype=np.int64)
    if len(labels) == 0:
        raise ValueError(f"{description} is empty")
    return labels


def _row(dataset, labels, task):
    if task == "binary":
        labels = np.where(labels == 0, 0, 1)
        class_names = ["negative", "positive"]
    else:
        class_names = FOUR_CLASS_NAMES
    metrics = calculate_chance_baselines(labels, len(class_names))
    return {
        "dataset": dataset,
        "task": task,
        "sample_count": metrics["sample_count"],
        "class_counts": json.dumps(
            {
                name: int(metrics["class_counts"][index])
                for index, name in enumerate(class_names)
            },
            sort_keys=True,
        ),
        "prevalence_weighted_expected_accuracy": metrics[
            "prevalence_weighted_expected_accuracy"
        ],
        "majority_class_accuracy": metrics["majority_class_accuracy"],
    }


def build_rows(external, target, source_validation):
    datasets = {
        "KPXKPR-50": _labels(
            external.loc[external["split"].eq("holdout")],
            "KPXKPR-50",
        ),
        "KPXKPR-64": _labels(external, "KPXKPR-64"),
        "Three-API-170": _labels(target, "Three-API-170"),
        "Source-validation": _labels(source_validation, "Source validation"),
    }
    return [
        _row(name, labels, task)
        for name, labels in datasets.items()
        for task in ["binary", "four-class"]
    ]


def _write_markdown(rows, path):
    lines = [
        "# Distribution-aware chance baselines",
        "",
        (
            "Prevalence-weighted expected accuracy is `sum(p_c^2)`, where "
            "`p_c` is the observed class prevalence. It represents a random "
            "predictor that samples labels from the same class distribution."
        ),
        "",
        "| Dataset | Task | N | Class counts | Weighted chance | Majority |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        counts = json.loads(row["class_counts"])
        counts_text = ", ".join(f"{key}={value}" for key, value in counts.items())
        lines.append(
            f"| {row['dataset']} | {row['task']} | {row['sample_count']} | "
            f"{counts_text} | "
            f"{row['prevalence_weighted_expected_accuracy']:.4f} | "
            f"{row['majority_class_accuracy']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    external = pd.read_csv(args.external_manifest, keep_default_na=False)
    target = pd.read_csv(args.target_table, keep_default_na=False)
    source_validation = pd.read_csv(
        args.source_validation_table,
        keep_default_na=False,
    )
    rows = build_rows(external, target, source_validation)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "chance_baselines.csv", index=False)
    payload = {
        "schema_version": "mcc-gcn-revision-chance-baselines-v1",
        "definition": "sum of squared observed class prevalences",
        "rows": rows,
    }
    (output_dir / "chance_baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(rows, output_dir / "chance_baselines.md")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Chance baselines ready: {output_dir}")


if __name__ == "__main__":
    main()

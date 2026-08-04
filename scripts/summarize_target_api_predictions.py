#!/usr/bin/env python
"""Combine target-API predictions and report pooled and per-API metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = {
    "binary": ["negative", "positive"],
    "four-class": [
        "negative",
        "salt",
        "cocrystal",
        "hydrate_or_solvate",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(CLASS_NAMES), required=True)
    parser.add_argument("--target-table", required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Evaluation CSV. Repeat for cross-validation folds.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow predictions for only a subset of the frozen target table.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_from_path(path: Path) -> int | None:
    match = re.search(r"fold-(\d+)", str(path))
    return int(match.group(1)) if match else None


def _load_predictions(paths: list[str]) -> pd.DataFrame:
    parts = []
    for value in paths:
        path = Path(value)
        table = pd.read_csv(path, keep_default_na=False)
        required = {"Pair Key", "True Label", "Predicted Label"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        table = table.rename(columns={"Pair Key": "pair_key"})
        table["prediction_file"] = str(path)
        table["test_fold"] = _fold_from_path(path)
        parts.append(table)
    combined = pd.concat(parts, ignore_index=True)
    duplicate = combined["pair_key"].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(
            "Prediction files contain duplicate physical pairs: "
            f"{combined.loc[duplicate, 'pair_key'].nunique()}"
        )
    return combined


def _balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_labels: list[int],
) -> float:
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    totals = matrix.sum(axis=1)
    present = totals > 0
    recalls = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(len(class_labels), dtype=float),
        where=present,
    )
    return float(recalls[present].mean())


def _bootstrap_intervals(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    class_labels: list[int],
    replicates: int,
    seed: int,
) -> dict:
    if replicates < 1:
        return {}
    rng = np.random.default_rng(seed)
    present_classes = sorted(set(labels.tolist()))
    indices_by_class = [
        np.flatnonzero(labels == label) for label in present_classes
    ]
    accuracy_values = []
    balanced_values = []
    for _ in range(replicates):
        sample = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in indices_by_class
            ]
        )
        accuracy_values.append(float(np.mean(labels[sample] == predictions[sample])))
        balanced_values.append(
            _balanced_accuracy(
                labels[sample],
                predictions[sample],
                class_labels,
            )
        )
    return {
        "method": "stratified_nonparametric_bootstrap",
        "replicates": replicates,
        "accuracy_95_ci": [
            float(value)
            for value in np.quantile(accuracy_values, [0.025, 0.975])
        ],
        "balanced_accuracy_95_ci": [
            float(value)
            for value in np.quantile(balanced_values, [0.025, 0.975])
        ],
    }


def _metrics(
    table: pd.DataFrame,
    class_names: list[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    labels = table["True Label"].to_numpy(dtype=np.int64)
    predictions = table["Predicted Label"].to_numpy(dtype=np.int64)
    class_labels = list(range(len(class_names)))
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    totals = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        totals,
        out=np.zeros(len(class_names), dtype=float),
        where=totals > 0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    present = totals > 0
    return {
        "pairs": len(table),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy_present_classes": float(recalls[present].mean()),
        "present_classes": [
            class_names[index]
            for index, is_present in enumerate(present)
            if is_present
        ],
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "support": int(support[index]),
                "accuracy": (
                    float(recalls[index]) if totals[index] else None
                ),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        },
        "bootstrap": _bootstrap_intervals(
            labels,
            predictions,
            class_labels=class_labels,
            replicates=bootstrap_replicates,
            seed=seed,
        ),
    }


def main():
    args = parse_args()
    class_names = CLASS_NAMES[args.task]
    target = pd.read_csv(args.target_table, keep_default_na=False)
    required = {
        "pair_key",
        "representative_pair_key",
        "label_int",
        "target_apis",
    }
    missing = required.difference(target.columns)
    if missing:
        raise ValueError(
            f"Target table is missing columns: {sorted(missing)}"
        )
    if target["pair_key"].nunique() != len(target):
        raise ValueError("Target table contains duplicate physical pairs")
    if target["representative_pair_key"].nunique() != len(target):
        raise ValueError(
            "Target table contains duplicate model-input physical pairs"
        )

    predictions = _load_predictions(args.prediction)
    prediction_keys = set(predictions["pair_key"])
    target_keys = set(target["representative_pair_key"])
    unexpected = prediction_keys.difference(target_keys)
    missing_keys = target_keys.difference(prediction_keys)
    if unexpected:
        raise ValueError(f"Predictions contain {len(unexpected)} unknown pairs")
    if missing_keys and not args.allow_incomplete:
        raise ValueError(f"Predictions are missing {len(missing_keys)} target pairs")

    metadata_columns = [
        "pair_key",
        "representative_pair_key",
        "reactant_A",
        "reactant_B",
        "label_str",
        "label_int",
        "identifier",
        "target_apis",
        "source_kind",
    ]
    metadata = target[metadata_columns].rename(
        columns={
            "pair_key": "connectivity_pair_key",
            "representative_pair_key": "pair_key",
        }
    )
    combined = predictions.merge(
        metadata,
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    expected_labels = (
        np.where(combined["label_int"].eq(0), 0, 1)
        if args.task == "binary"
        else combined["label_int"].to_numpy(dtype=np.int64)
    )
    if not np.array_equal(
        combined["True Label"].to_numpy(dtype=np.int64),
        expected_labels,
    ):
        raise ValueError("Prediction labels disagree with the target table")

    metrics = {
        "schema_version": "mcc-gcn-three-api-prediction-summary-v1",
        "task": args.task,
        "class_names": class_names,
        "pooled": _metrics(
            combined,
            class_names,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        "by_target_api": {},
        "inputs": {
            "target_table": {
                "path": args.target_table,
                "sha256": sha256_file(args.target_table),
            },
            "prediction_files": [
                {"path": path, "sha256": sha256_file(path)}
                for path in args.prediction
            ],
        },
    }
    api_names = sorted(
        {
            name
            for values in target["target_apis"].str.split(";")
            for name in values
            if name
        }
    )
    for index, name in enumerate(api_names):
        subset = combined.loc[
            combined["target_apis"].str.split(";").map(
                lambda names: name in names
            )
        ]
        metrics["by_target_api"][name] = _metrics(
            subset,
            class_names,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed + index + 1,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "combined_predictions.csv"
    combined.sort_values(["test_fold", "pair_key"], na_position="first").to_csv(
        predictions_path,
        index=False,
    )
    metrics["combined_predictions"] = {
        "path": str(predictions_path),
        "sha256": sha256_file(predictions_path),
    }
    metrics_path = output_dir / "summary_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

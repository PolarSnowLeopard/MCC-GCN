#!/usr/bin/env python3
"""Freeze task-specific undersampled source tables for the three-API study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-physical", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(table):
    required = {"reactant_A", "reactant_B", "label_int", "pair_key"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Training table is missing {sorted(missing)}")
    if table.empty or table["pair_key"].duplicated().any():
        raise ValueError("Training physical pairs must be non-empty and unique")
    labels = pd.to_numeric(table["label_int"], errors="raise").astype(int)
    if set(labels) != {0, 1, 2, 3}:
        raise ValueError("Training table must contain all four classes")


def _sample_groups(table, group_labels, seed):
    counts = group_labels.value_counts().sort_index()
    sample_size = int(counts.min())
    selected = []
    for group in counts.index:
        group_table = table.loc[group_labels.eq(group)]
        selected.append(
            group_table.sample(
                n=sample_size,
                replace=False,
                random_state=seed + int(group),
            )
        )
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values("pair_key", kind="stable")
        .reset_index(drop=True)
    )


def build_undersampled_tables(table, seed):
    _validate(table)
    labels = pd.to_numeric(table["label_int"], errors="raise").astype(int)
    four_class = _sample_groups(table, labels, seed)
    binary_labels = pd.Series(
        np.where(labels.eq(0), 0, 1),
        index=table.index,
        dtype=np.int64,
    )
    binary = _sample_groups(table, binary_labels, seed)
    binary["binary_label_int"] = np.where(binary["label_int"].eq(0), 0, 1)
    return {"binary": binary, "four-class": four_class}


def augment_pair_orders(table):
    forward = table.copy()
    forward["pair_order"] = "A_B"
    reverse = table.copy()
    reverse[["reactant_A", "reactant_B"]] = reverse[
        ["reactant_B", "reactant_A"]
    ].to_numpy()
    reverse["pair_order"] = "B_A"
    return pd.concat([forward, reverse], ignore_index=True)


def _counts(table, task):
    labels = table["label_int"].astype(int).to_numpy()
    if task == "binary":
        labels = np.where(labels == 0, 0, 1)
        names = ["negative", "positive"]
    else:
        names = ["negative", "salt", "cocrystal", "hydrate_or_solvate"]
    values = np.bincount(labels, minlength=len(names))
    return {name: int(values[index]) for index, name in enumerate(names)}


def main():
    args = parse_args()
    source = Path(args.train_physical)
    table = pd.read_csv(source, keep_default_na=False)
    subsets = build_undersampled_tables(table, args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for task, subset in subsets.items():
        task_key = task.replace("-", "_")
        physical_path = output_dir / f"{task_key}_undersampled_physical.csv"
        ordered_path = output_dir / f"{task_key}_undersampled_ordered.csv"
        subset.to_csv(physical_path, index=False)
        augment_pair_orders(subset).to_csv(ordered_path, index=False)
        outputs[task] = {
            "physical_pairs": len(subset),
            "ordered_rows": len(subset) * 2,
            "class_counts": _counts(subset, task),
            "physical_table": {
                "path": str(physical_path),
                "sha256": _sha256(physical_path),
            },
            "ordered_table": {
                "path": str(ordered_path),
                "sha256": _sha256(ordered_path),
            },
        }
    manifest = {
        "schema_version": "mcc-gcn-target-pretrain-ablation-v1",
        "strategy": "task_specific_random_undersampling_before_order_augmentation",
        "subset_seed": args.seed,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "physical_pairs": len(table),
        },
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

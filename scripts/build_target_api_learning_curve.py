#!/usr/bin/env python3
"""Freeze nested, class-balanced fine-tuning subsets for every outer fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.learning_curve import balanced_nested_subsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sizes", default="8,16,32,48")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    sizes = sorted(
        set(
            int(value.strip())
            for value in args.sizes.split(",")
            if value.strip()
        )
    )
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    outputs = {}
    fold_counts = {}

    for fold in range(args.folds):
        source = (
            data_root
            / "folds"
            / f"fold-{fold}"
            / "train_physical_pairs.csv"
        )
        table = pd.read_csv(source, keep_default_na=False)
        subsets = balanced_nested_subsets(table, sizes, seed=args.seed)
        fold_counts[str(fold)] = {}
        for size, subset in subsets.items():
            path = output_root / f"fold-{fold}" / f"size-{size}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(path, index=False)
            outputs[str(path.relative_to(output_root))] = sha256_file(path)
            fold_counts[str(fold)][str(size)] = {
                "physical_pairs": len(subset),
                "class_counts": {
                    str(int(label)): int(count)
                    for label, count in (
                        subset["label_int"]
                        .value_counts()
                        .sort_index()
                        .items()
                    )
                },
            }

    manifest = {
        "schema_version": "mcc-gcn-target-api-learning-curve-v1",
        "strategy": "nested_class_balanced_physical_pairs",
        "subset_seed": args.seed,
        "sizes": sizes,
        "folds": args.folds,
        "fold_counts": fold_counts,
        "outputs": outputs,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

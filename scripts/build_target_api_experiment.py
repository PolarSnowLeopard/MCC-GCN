#!/usr/bin/env python
"""Freeze the three-API benchmark, exclusions, and outer CV folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import rdBase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.quality import read_pair_table
from mcc_gcn.data.splitting import grouped_stratified_split
from mcc_gcn.data.target_api import (
    TARGET_APIS,
    curate_target_api_experiment,
    stratified_target_folds,
)


EXPECTED_TARGET_COUNTS = {
    "negative:0": 27,
    "salt:1": 17,
    "cocrystal:2": 106,
    "hydrate_or_solvate:3": 20,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csd-pairs", required=True)
    parser.add_argument("--negative-pairs", required=True)
    parser.add_argument("--pretrain-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--skip-expected-count-check",
        action="store_true",
        help="Allow non-frozen source data for development only.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def augment_pair_orders(table: pd.DataFrame) -> pd.DataFrame:
    forward = table.copy()
    forward["pair_order"] = "A_B"
    reverse = table.copy()
    reverse[["reactant_A", "reactant_B"]] = reverse[
        ["reactant_B", "reactant_A"]
    ]
    reverse["pair_order"] = "B_A"
    return pd.concat([forward, reverse], ignore_index=True)


def _write_table(path: Path, table: pd.DataFrame, hashes: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    hashes[str(path)] = sha256_file(path)


def _label_counts(table: pd.DataFrame) -> dict[str, int]:
    counts = table.groupby(["label_str", "label_int"]).size().sort_index()
    return {
        f"{label}:{int(label_int)}": int(count)
        for (label, label_int), count in counts.items()
    }


def _target_api_counts(table: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for target in TARGET_APIS:
        mask = table["target_apis"].str.split(";").map(
            lambda names: target.name in names
        )
        counts[target.name] = int(mask.sum())
    return counts


def _build_pretrain_split(table, validation_fraction, seed):
    split = grouped_stratified_split(
        table["label_int"].to_numpy(),
        table["pair_key"].to_numpy(),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    assignment = dict(zip(split.manifest["pair_key"], split.manifest["split"]))
    with_split = table.copy()
    with_split["split"] = with_split["pair_key"].map(assignment)
    train = with_split.loc[with_split["split"].eq("train")].copy()
    validation = with_split.loc[
        with_split["split"].eq("validation")
    ].copy()
    return with_split, train, validation, split.manifest


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    csd = read_pair_table(args.csd_pairs)
    negative = read_pair_table(args.negative_pairs)
    pretrain = pd.read_csv(args.pretrain_pairs, keep_default_na=False)
    result = curate_target_api_experiment(csd, negative, pretrain)

    target_counts = _label_counts(result.four_class_pairs)
    if not args.skip_expected_count_check:
        if target_counts != EXPECTED_TARGET_COUNTS:
            raise ValueError(
                "Frozen target counts changed: "
                f"expected {EXPECTED_TARGET_COUNTS}, got {target_counts}"
            )
        conflict_pairs = result.conflicts["connectivity_pair_key"].nunique()
        if conflict_pairs != 6:
            raise ValueError(
                f"Expected 6 quarantined target conflicts, got {conflict_pairs}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    root_outputs = {
        "target_evidence.csv": result.evidence,
        "target_conflicts.csv": result.conflicts,
        "four_class_target_physical_pairs.csv": result.four_class_pairs,
        "binary_target_physical_pairs.csv": result.binary_pairs,
        "pretrain_target_api_exclusions.csv": result.pretrain_exclusions,
        "four_class_pretrain_excluding_target_apis.csv": (
            result.four_class_pretrain_pairs
        ),
        "binary_pretrain_excluding_target_apis.csv": (
            result.binary_pretrain_pairs
        ),
    }
    for name, table in root_outputs.items():
        _write_table(output_dir / name, table, hashes)

    (
        pretrain_with_split,
        pretrain_train,
        pretrain_validation,
        pretrain_split_manifest,
    ) = _build_pretrain_split(
        result.four_class_pretrain_pairs,
        args.pretrain_validation_fraction,
        args.seed,
    )
    pretrain_dir = output_dir / "pretrain-split"
    pretrain_outputs = {
        "physical_pairs_with_split.csv": pretrain_with_split,
        "train_physical_pairs.csv": pretrain_train,
        "validation_physical_pairs.csv": pretrain_validation,
        "train_ordered_pairs.csv": augment_pair_orders(pretrain_train),
        "validation_ordered_pairs.csv": augment_pair_orders(
            pretrain_validation
        ),
        "split_manifest.csv": pretrain_split_manifest,
    }
    for name, table in pretrain_outputs.items():
        _write_table(pretrain_dir / name, table, hashes)

    fold_assignments = stratified_target_folds(
        result.four_class_pairs,
        n_splits=args.folds,
        seed=args.seed,
    )
    folds_dir = output_dir / "folds"
    _write_table(folds_dir / "fold_assignments.csv", fold_assignments, hashes)
    fold_counts = {}
    for fold in range(args.folds):
        test_keys = set(
            fold_assignments.loc[
                fold_assignments["test_fold"].eq(fold),
                "pair_key",
            ]
        )
        test = result.four_class_pairs.loc[
            result.four_class_pairs["pair_key"].isin(test_keys)
        ].copy()
        train = result.four_class_pairs.loc[
            ~result.four_class_pairs["pair_key"].isin(test_keys)
        ].copy()
        if set(train["pair_key"]).intersection(test["pair_key"]):
            raise AssertionError(f"Pair overlap in outer fold {fold}")
        fold_dir = folds_dir / f"fold-{fold}"
        _write_table(fold_dir / "train_physical_pairs.csv", train, hashes)
        _write_table(fold_dir / "test_physical_pairs.csv", test, hashes)
        fold_counts[str(fold)] = {
            "train_pairs": len(train),
            "test_pairs": len(test),
            "test_coformer_groups": int(
                fold_assignments.loc[
                    fold_assignments["test_fold"].eq(fold),
                    "coformer_connectivity_key",
                ].nunique()
            ),
            "test_labels": _label_counts(test),
            "test_target_apis": _target_api_counts(test),
        }

    relative_hashes = {
        str(Path(path).relative_to(output_dir)): digest
        for path, digest in hashes.items()
    }
    manifest = {
        "schema_version": "mcc-gcn-three-api-experiment-v2",
        "software": {
            "pandas": pd.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "targets": [
            {
                "name": target.name,
                "cas_number": target.cas_number,
                "inchi_connectivity_key": target.connectivity_key,
            }
            for target in TARGET_APIS
        ],
        "inputs": {
            name: {"path": str(Path(path)), "sha256": sha256_file(path)}
            for name, path in {
                "csd_pairs": args.csd_pairs,
                "negative_pairs": args.negative_pairs,
                "pretrain_pairs": args.pretrain_pairs,
            }.items()
        },
        "target_benchmark": {
            "physical_pairs": len(result.four_class_pairs),
            "label_counts": target_counts,
            "target_api_counts": _target_api_counts(
                result.four_class_pairs
            ),
            "quarantined_conflict_pairs": int(
                result.conflicts["connectivity_pair_key"].nunique()
            ),
        },
        "pretraining": {
            "source_physical_pairs": len(pretrain),
            "excluded_target_api_pairs": len(result.pretrain_exclusions),
            "remaining_physical_pairs": len(
                result.four_class_pretrain_pairs
            ),
            "validation_fraction": args.pretrain_validation_fraction,
            "train_pairs": len(pretrain_train),
            "validation_pairs": len(pretrain_validation),
        },
        "outer_cross_validation": {
            "folds": args.folds,
            "seed": args.seed,
            "strategy": "stratified_group_k_fold",
            "group_by": "coformer_connectivity_key",
            "coformer_groups": int(
                fold_assignments["coformer_connectivity_key"].nunique()
            ),
            "coformer_groups_spanning_folds": int(
                fold_assignments.groupby("coformer_connectivity_key")[
                    "test_fold"
                ].nunique().gt(1).sum()
            ),
            "fold_counts": fold_counts,
        },
        "outputs": relative_hashes,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

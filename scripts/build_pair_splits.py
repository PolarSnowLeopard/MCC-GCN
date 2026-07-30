#!/usr/bin/env python
"""Freeze train/validation physical-pair splits before augmentation."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.splitting import grouped_stratified_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-split-manifest",
        help=(
            "Reuse pair assignments from another task table. The new table "
            "may be a strict subset after additional quality filtering."
        ),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def augment_pair_orders(table):
    forward = table.copy()
    forward["pair_order"] = "A_B"
    reverse = table.copy()
    reverse[["reactant_A", "reactant_B"]] = reverse[
        ["reactant_B", "reactant_A"]
    ]
    reverse["pair_order"] = "B_A"
    return pd.concat([forward, reverse], ignore_index=True)


def main():
    args = parse_args()
    table = pd.read_csv(args.input, keep_default_na=False)
    required = {
        "reactant_A",
        "reactant_B",
        "label_str",
        "label_int",
        "identifier",
        "pair_key",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if table["pair_key"].nunique() != len(table):
        raise ValueError("Input must contain one row per physical pair")

    if args.reuse_split_manifest:
        reused = pd.read_csv(
            args.reuse_split_manifest,
            keep_default_na=False,
        )
        missing_assignments = set(table["pair_key"]).difference(
            reused["pair_key"]
        )
        if missing_assignments:
            raise ValueError(
                "Reused split manifest is missing "
                f"{len(missing_assignments)} current pair keys"
            )
        reused_seeds = set(reused["seed"])
        reused_fractions = set(reused["validation_fraction"])
        if reused_seeds != {args.seed}:
            raise ValueError("Reused split manifest seed differs from --seed")
        if reused_fractions != {args.validation_fraction}:
            raise ValueError(
                "Reused split validation fraction differs from argument"
            )
        assignment = dict(zip(reused["pair_key"], reused["split"]))
        split_manifest = table[["pair_key", "label_int"]].copy()
        split_manifest["split"] = split_manifest["pair_key"].map(assignment)
        split_manifest["augmented_row_count"] = 1
        split_manifest["seed"] = args.seed
        split_manifest["validation_fraction"] = args.validation_fraction
        split_manifest = split_manifest.sort_values(
            ["split", "label_int", "pair_key"]
        )
        split_source = {
            "path": str(Path(args.reuse_split_manifest)),
            "sha256": sha256_file(args.reuse_split_manifest),
            "filtered_pair_keys": int(
                len(set(reused["pair_key"]).difference(table["pair_key"]))
            ),
        }
    else:
        split = grouped_stratified_split(
            table["label_int"].to_numpy(),
            table["pair_key"].to_numpy(),
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        assignment = dict(
            zip(split.manifest["pair_key"], split.manifest["split"])
        )
        split_manifest = split.manifest
        split_source = None
    table["split"] = table["pair_key"].map(assignment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_physical = table.loc[table["split"].eq("train")]
    validation_physical = table.loc[table["split"].eq("validation")]
    outputs = {
        "physical_pairs_with_split.csv": table,
        "train_physical_pairs.csv": train_physical,
        "validation_physical_pairs.csv": validation_physical,
        "train_ordered_pairs.csv": augment_pair_orders(train_physical),
        "validation_ordered_pairs.csv": augment_pair_orders(
            validation_physical
        ),
        "split_manifest.csv": split_manifest,
    }
    hashes = {}
    for filename, output in outputs.items():
        path = output_dir / filename
        output.to_csv(path, index=False)
        hashes[filename] = sha256_file(path)

    counts = (
        table.groupby(["split", "label_str", "label_int"])
        .size()
        .sort_index()
    )
    manifest = {
        "schema_version": "mcc-gcn-physical-pair-split-v1",
        "input": {
            "path": str(Path(args.input)),
            "sha256": sha256_file(args.input),
        },
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "physical_pairs": len(table),
        "pair_overlap": 0,
        "reused_split_manifest": split_source,
        "counts": {
            f"{split_name}:{label}:{label_int}": int(count)
            for (split_name, label, label_int), count in counts.items()
        },
        "outputs": hashes,
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

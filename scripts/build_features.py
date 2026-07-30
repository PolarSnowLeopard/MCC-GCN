#!/usr/bin/env python
"""Build versioned MCC-GCN graph features from an ordered pair table."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import rdBase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.dataset import GraphDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--feature-source",
        choices=["rdkit_smiles", "ccdc_molblock"],
        default="rdkit_smiles",
    )
    parser.add_argument("--mol-blocks")
    parser.add_argument(
        "--adjacency-type",
        default="OnlyCovalentBond",
    )
    parser.add_argument("--hbond", action="store_true")
    parser.add_argument("--pipi-stack", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument(
        "--rdkit-coordinate-mode",
        choices=["2d", "3d"],
        default="2d",
    )
    parser.add_argument("--max-graph-size", type=int)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rejections(path, rejections):
    fieldnames = [
        "reactant_A",
        "reactant_B",
        "label_str",
        "label_int",
        "identifier",
        "error_type",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejections)


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_path = output_path.with_suffix(".rejections.csv")
    manifest_path = output_path.with_suffix(".manifest.json")

    dataset = GraphDataset(
        args.input,
        args.mol_blocks,
        feature_source=args.feature_source,
        rdkit_coordinate_mode=args.rdkit_coordinate_mode,
    )
    try:
        dataset.make_graph_dataset(
            A_type=args.adjacency_type,
            hbond=args.hbond,
            pipi_stack=args.pipi_stack,
            contact=args.contact,
            max_graph_size=args.max_graph_size,
            save_name=output_path,
            strict=True,
        )
    except ValueError:
        write_rejections(rejection_path, dataset.rejections)
        raise
    write_rejections(rejection_path, [])

    features = np.load(output_path, allow_pickle=False)
    pair_keys = features["pair_keys"]
    manifest = {
        "schema_version": "mcc-gcn-graph-features-v1",
        "rdkit_version": rdBase.rdkitVersion,
        "input": {
            "path": str(Path(args.input)),
            "sha256": sha256_file(args.input),
        },
        "configuration": {
            "feature_source": args.feature_source,
            "mol_blocks": (
                str(Path(args.mol_blocks)) if args.mol_blocks else None
            ),
            "adjacency_type": args.adjacency_type,
            "hbond": args.hbond,
            "pipi_stack": args.pipi_stack,
            "contact": args.contact,
            "rdkit_coordinate_mode": args.rdkit_coordinate_mode,
            "max_graph_size": args.max_graph_size,
        },
        "ordered_rows": int(len(features["labels"])),
        "physical_pairs": int(len(set(pair_keys.astype(str)))),
        "max_observed_graph_size": int(features["graph_size"].max()),
        "padded_width": int(features["V"].shape[1]),
        "node_feature_width": int(features["V"].shape[2]),
        "adjacency_channels": int(features["A"].shape[2]),
        "label_counts": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(features["labels"], return_counts=True)
            )
        },
        "outputs": {
            output_path.name: sha256_file(output_path),
            rejection_path.name: sha256_file(rejection_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

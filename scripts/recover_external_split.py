#!/usr/bin/env python
"""Recover the historical 14/50 external split from frozen NPZ artifacts."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.lineage import (
    build_unique_fingerprint_index,
    match_feature_subset,
    sample_feature_fingerprint,
)
from mcc_gcn.data.quality import (
    audit_smiles,
    canonical_pair_key,
    read_pair_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--reference-forward", required=True)
    parser.add_argument("--reference-reverse", required=True)
    parser.add_argument("--holdout-forward", required=True)
    parser.add_argument("--fine-tune", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--source-ref",
        required=True,
        help="Immutable repository commit and source path.",
    )
    parser.add_argument("--reference-forward-ref", required=True)
    parser.add_argument("--reference-reverse-ref", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path):
    return np.load(path, allow_pickle=False)


def canonicalize_pair(smiles_a, smiles_b):
    molecule_a = audit_smiles(smiles_a)
    molecule_b = audit_smiles(smiles_b)
    if molecule_a.error or molecule_b.error:
        raise ValueError(
            f"Invalid external pair: A={molecule_a.error}, "
            f"B={molecule_b.error}"
        )
    return canonical_pair_key(
        molecule_a.canonical_smiles,
        molecule_b.canonical_smiles,
    )


def main():
    args = parse_args()
    source = read_pair_table(args.source_csv)
    reference_forward = load_npz(args.reference_forward)
    reference_reverse = load_npz(args.reference_reverse)
    holdout_forward = load_npz(args.holdout_forward)
    fine_tune = load_npz(args.fine_tune)

    reference_count = len(reference_forward["labels"])
    if len(source) != reference_count:
        raise ValueError(
            f"Source rows ({len(source)}) != reference rows "
            f"({reference_count})"
        )
    if len(reference_reverse["labels"]) != reference_count:
        raise ValueError("Forward and reverse references differ in length")
    if not np.array_equal(
        reference_forward["labels"],
        reference_reverse["labels"],
    ):
        raise ValueError("Forward and reverse reference labels differ")

    holdout_indices = match_feature_subset(
        reference_forward,
        holdout_forward,
    )
    fine_tune_indices = sorted(
        set(range(reference_count)).difference(holdout_indices)
    )

    fine_tune_feature_index = build_unique_fingerprint_index(fine_tune)
    fine_tune_forward_positions = {}
    fine_tune_reverse_positions = {}
    for reference_index in fine_tune_indices:
        forward_fingerprint = sample_feature_fingerprint(
            reference_forward,
            reference_index,
        )
        reverse_fingerprint = sample_feature_fingerprint(
            reference_reverse,
            reference_index,
        )
        if forward_fingerprint not in fine_tune_feature_index:
            raise ValueError(
                f"Fine-tune artifact lacks forward sample {reference_index}"
            )
        if reverse_fingerprint not in fine_tune_feature_index:
            raise ValueError(
                f"Fine-tune artifact lacks reverse sample {reference_index}"
            )
        fine_tune_forward_positions[reference_index] = (
            fine_tune_feature_index[forward_fingerprint]
        )
        fine_tune_reverse_positions[reference_index] = (
            fine_tune_feature_index[reverse_fingerprint]
        )

    holdout_positions = {
        reference_index: holdout_index
        for holdout_index, reference_index in enumerate(holdout_indices)
    }
    rows = []
    for index, row in enumerate(source.itertuples(index=False)):
        label = int(row.label_int)
        if label != int(reference_forward["labels"][index]):
            raise ValueError(f"Label mismatch at external index {index}")
        split = "fine_tune" if index in fine_tune_indices else "holdout"
        rows.append(
            {
                "external_index": index,
                "source_row": row.source_row,
                "reactant_A": row.reactant_A,
                "reactant_B": row.reactant_B,
                "label_str": row.label_str,
                "label_int": label,
                "identifier": row.identifier,
                "pair_key": canonicalize_pair(
                    row.reactant_A,
                    row.reactant_B,
                ),
                "split": split,
                "holdout_feature_index": holdout_positions.get(index, ""),
                "fine_tune_forward_feature_index": (
                    fine_tune_forward_positions.get(index, "")
                ),
                "fine_tune_reverse_feature_index": (
                    fine_tune_reverse_positions.get(index, "")
                ),
                "reference_forward_fingerprint": (
                    sample_feature_fingerprint(reference_forward, index)
                ),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest["pair_key"].nunique() != len(manifest):
        raise ValueError("External source contains duplicate unordered pairs")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False)

    class_counts = (
        manifest.groupby(["split", "label_str", "label_int"])
        .size()
        .sort_index()
    )
    metadata = {
        "schema_version": "mcc-gcn-external-split-lineage-v1",
        "source_ref": args.source_ref,
        "inputs": {
            name: {
                "path": str(Path(path)),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "source_csv": args.source_csv,
                "reference_forward": args.reference_forward,
                "reference_reverse": args.reference_reverse,
                "holdout_forward": args.holdout_forward,
                "fine_tune": args.fine_tune,
            }.items()
        },
        "external_pairs": len(manifest),
        "fine_tune_indices": fine_tune_indices,
        "holdout_indices": holdout_indices,
        "class_counts": {
            f"{split}:{label}:{label_int}": int(count)
            for (split, label, label_int), count in class_counts.items()
        },
        "output_csv": {
            "path": str(Path(args.output_csv)),
            "sha256": sha256_file(args.output_csv),
        },
    }
    metadata["inputs"]["source_csv"]["ref"] = args.source_ref
    metadata["inputs"]["reference_forward"]["ref"] = (
        args.reference_forward_ref
    )
    metadata["inputs"]["reference_reverse"]["ref"] = (
        args.reference_reverse_ref
    )
    Path(args.output_json).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

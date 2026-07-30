#!/usr/bin/env python
"""Freeze the 20 Minoxidil and 14 external fine-tuning physical pairs."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.quality import (
    audit_smiles,
    canonical_pair_key,
    read_pair_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minoxidil-ordered-csv", required=True)
    parser.add_argument("--external-split-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ccdc-reexport-manifest", required=True)
    parser.add_argument("--minoxidil-source-ref", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_pair(smiles_a, smiles_b):
    molecule_a = audit_smiles(smiles_a)
    molecule_b = audit_smiles(smiles_b)
    if molecule_a.error or molecule_b.error:
        raise ValueError(
            f"Invalid locked pair: A={molecule_a.error}, "
            f"B={molecule_b.error}"
        )
    return canonical_pair_key(
        molecule_a.canonical_smiles,
        molecule_b.canonical_smiles,
    )


def main():
    args = parse_args()
    minoxidil = read_pair_table(args.minoxidil_ordered_csv)
    if len(minoxidil) != 40:
        raise ValueError("Expected 40 ordered Minoxidil rows")

    forward = minoxidil.iloc[:20].reset_index(drop=True)
    reverse = minoxidil.iloc[20:].reset_index(drop=True)
    for index in range(20):
        forward_pair = sorted(
            [
                forward.loc[index, "reactant_A"],
                forward.loc[index, "reactant_B"],
            ]
        )
        reverse_pair = sorted(
            [
                reverse.loc[index, "reactant_A"],
                reverse.loc[index, "reactant_B"],
            ]
        )
        if forward_pair != reverse_pair:
            raise ValueError(
                f"Minoxidil reverse row mismatch at physical pair {index}"
            )

    rows = []
    for index, row in enumerate(forward.itertuples(index=False)):
        has_refcode = bool(row.identifier)
        rows.append(
            {
                "lock_id": f"minoxidil-{index:02d}",
                "source_dataset": "minoxidil_20",
                "source_index": index,
                "reactant_A_legacy": row.reactant_A,
                "reactant_B_legacy": row.reactant_B,
                "label_str": row.label_str,
                "label_int": int(row.label_int),
                "identifier": row.identifier,
                "model_pair_key": (
                    ""
                    if has_refcode
                    else canonical_pair(row.reactant_A, row.reactant_B)
                ),
                "structure_status": (
                    "requires_ccdc_reexport"
                    if has_refcode
                    else "ready_rdkit"
                ),
            }
        )

    external = pd.read_csv(
        args.external_split_manifest,
        keep_default_na=False,
    )
    external = external.loc[external["split"].eq("fine_tune")]
    if len(external) != 14:
        raise ValueError("Expected 14 external fine-tuning pairs")
    for row in external.itertuples(index=False):
        rows.append(
            {
                "lock_id": f"external-{int(row.external_index):02d}",
                "source_dataset": "external_64",
                "source_index": int(row.external_index),
                "reactant_A_legacy": row.reactant_A,
                "reactant_B_legacy": row.reactant_B,
                "label_str": row.label_str,
                "label_int": int(row.label_int),
                "identifier": row.identifier,
                "model_pair_key": row.pair_key,
                "structure_status": "ready_rdkit",
            }
        )

    manifest = pd.DataFrame(rows)
    if len(manifest) != 34:
        raise AssertionError("Fine-tuning lock must contain 34 physical pairs")
    if manifest["lock_id"].nunique() != len(manifest):
        raise AssertionError("Fine-tuning lock IDs are not unique")

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False)
    ccdc_reexport = manifest.loc[
        manifest["structure_status"].eq("requires_ccdc_reexport"),
        ["identifier", "label_str", "label_int", "lock_id"],
    ]
    ccdc_reexport.to_csv(args.ccdc_reexport_manifest, index=False)
    counts = (
        manifest.groupby(
            ["source_dataset", "structure_status", "label_str", "label_int"]
        )
        .size()
        .sort_index()
    )
    metadata = {
        "schema_version": "mcc-gcn-finetune-lock-v1",
        "minoxidil_source_ref": args.minoxidil_source_ref,
        "inputs": {
            "minoxidil_ordered_csv": {
                "path": str(Path(args.minoxidil_ordered_csv)),
                "sha256": sha256_file(args.minoxidil_ordered_csv),
            },
            "external_split_manifest": {
                "path": str(Path(args.external_split_manifest)),
                "sha256": sha256_file(args.external_split_manifest),
            },
        },
        "physical_pairs": len(manifest),
        "requires_ccdc_reexport": int(
            manifest["structure_status"]
            .eq("requires_ccdc_reexport")
            .sum()
        ),
        "counts": {
            f"{source}:{status}:{label}:{label_int}": int(count)
            for (source, status, label, label_int), count in counts.items()
        },
        "output_csv": {
            "path": str(Path(args.output_csv)),
            "sha256": sha256_file(args.output_csv),
        },
        "ccdc_reexport_manifest": {
            "path": str(Path(args.ccdc_reexport_manifest)),
            "sha256": sha256_file(args.ccdc_reexport_manifest),
        },
    }
    Path(args.output_json).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

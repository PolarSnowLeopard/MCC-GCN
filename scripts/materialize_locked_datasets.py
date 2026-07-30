#!/usr/bin/env python
"""Materialize final or explicitly provisional locked evaluation tables."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.quality import audit_smiles, canonical_pair_key

PAIR_COLUMNS = [
    "reactant_A",
    "reactant_B",
    "label_str",
    "label_int",
    "identifier",
    "pair_key",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune-lock", required=True)
    parser.add_argument("--external-split-lock", required=True)
    parser.add_argument(
        "--approved-minoxidil-pairs",
        help=(
            "Human-reviewed CSV with lock_id, reactant_A, reactant_B, and "
            "review_status=approved for all CCDC-dependent Minoxidil rows. "
            "Required in final mode and forbidden in provisional-no-ccdc mode."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["final", "provisional-no-ccdc"],
        default="final",
        help=(
            "Final mode requires all chemistry approvals. Provisional mode "
            "excludes unresolved rows and marks its outputs as non-final."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_pair(smiles_a, smiles_b, description):
    audited_a = audit_smiles(smiles_a)
    audited_b = audit_smiles(smiles_b)
    if audited_a.error or audited_b.error:
        raise ValueError(
            f"{description} has invalid SMILES: "
            f"A={audited_a.error}, B={audited_b.error}"
        )
    if audited_a.canonical_smiles == audited_b.canonical_smiles:
        raise ValueError(f"{description} contains two identical components")
    return (
        audited_a.canonical_smiles,
        audited_b.canonical_smiles,
        canonical_pair_key(
            audited_a.canonical_smiles,
            audited_b.canonical_smiles,
        ),
    )


def validate_unique_pairs(table, description):
    duplicated = table["pair_key"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"{description} contains "
            f"{table.loc[duplicated, 'pair_key'].nunique()} duplicate pairs"
        )


def binary_view(table):
    result = table.copy()
    result["label_str"] = result["label_int"].map(
        lambda value: "negative" if int(value) == 0 else "positive"
    )
    result["label_int"] = result["label_int"].map(
        lambda value: 0 if int(value) == 0 else 1
    )
    return result


def main():
    args = parse_args()
    fine_tune_lock = pd.read_csv(
        args.finetune_lock,
        keep_default_na=False,
    )
    external_lock = pd.read_csv(
        args.external_split_lock,
        keep_default_na=False,
    )
    required_approval_columns = {
        "lock_id",
        "reactant_A",
        "reactant_B",
        "review_status",
    }
    if args.mode == "final":
        if not args.approved_minoxidil_pairs:
            raise ValueError(
                "--approved-minoxidil-pairs is required in final mode"
            )
        approved = pd.read_csv(
            args.approved_minoxidil_pairs,
            keep_default_na=False,
        )
        missing = required_approval_columns.difference(approved.columns)
        if missing:
            raise ValueError(
                "Approved Minoxidil CSV is missing columns: "
                f"{sorted(missing)}"
            )
        if approved["lock_id"].duplicated().any():
            raise ValueError(
                "Approved Minoxidil lock_id values are not unique"
            )
        not_approved = approved.loc[
            ~approved["review_status"].eq("approved"),
            "lock_id",
        ]
        if not not_approved.empty:
            raise ValueError(
                "Every supplied Minoxidil pair must have "
                "review_status=approved: "
                + ", ".join(not_approved)
            )
    else:
        if args.approved_minoxidil_pairs:
            raise ValueError(
                "--approved-minoxidil-pairs cannot be used in "
                "provisional-no-ccdc mode"
            )
        approved = pd.DataFrame(columns=sorted(required_approval_columns))
    approved_by_id = approved.set_index("lock_id")

    fine_tune_rows = []
    minoxidil_rows = []
    required_locks = fine_tune_lock.loc[
        fine_tune_lock["structure_status"].eq("requires_ccdc_reexport"),
        "lock_id",
    ]
    excluded_locks = []
    if args.mode == "final":
        missing_approvals = sorted(
            set(required_locks).difference(approved_by_id.index)
        )
        extra_approvals = sorted(
            set(approved_by_id.index).difference(required_locks)
        )
        if missing_approvals or extra_approvals:
            raise ValueError(
                "Approved Minoxidil lock IDs do not exactly match the "
                f"required set; missing={missing_approvals}, "
                f"extra={extra_approvals}"
            )
    else:
        excluded_locks = sorted(required_locks)

    for row in fine_tune_lock.itertuples(index=False):
        if row.structure_status == "requires_ccdc_reexport":
            if args.mode == "provisional-no-ccdc":
                continue
            approved_row = approved_by_id.loc[row.lock_id]
            smiles_a = approved_row["reactant_A"]
            smiles_b = approved_row["reactant_B"]
        elif row.structure_status == "ready_rdkit":
            smiles_a = row.reactant_A_legacy
            smiles_b = row.reactant_B_legacy
        else:
            raise ValueError(
                f"Unknown structure_status for {row.lock_id}: "
                f"{row.structure_status}"
            )
        canonical_a, canonical_b, pair_key = canonicalize_pair(
            smiles_a,
            smiles_b,
            row.lock_id,
        )
        materialized = {
            "reactant_A": canonical_a,
            "reactant_B": canonical_b,
            "label_str": row.label_str,
            "label_int": int(row.label_int),
            "identifier": row.identifier,
            "pair_key": pair_key,
            "lock_id": row.lock_id,
            "source_dataset": row.source_dataset,
        }
        fine_tune_rows.append(materialized)
        if row.source_dataset == "minoxidil_20":
            minoxidil_rows.append(materialized)

    fine_tune = pd.DataFrame(fine_tune_rows)
    minoxidil = pd.DataFrame(minoxidil_rows)
    external_rows = []
    for row in external_lock.itertuples(index=False):
        canonical_a, canonical_b, pair_key = canonicalize_pair(
            row.reactant_A,
            row.reactant_B,
            f"external-{int(row.external_index):02d}",
        )
        if pair_key != row.pair_key:
            raise ValueError(
                f"External pair-key drift at index {row.external_index}"
            )
        external_rows.append(
            {
                "reactant_A": canonical_a,
                "reactant_B": canonical_b,
                "label_str": row.label_str,
                "label_int": int(row.label_int),
                "identifier": row.identifier,
                "pair_key": pair_key,
                "lock_id": f"external-{int(row.external_index):02d}",
                "source_dataset": "external_64",
                "external_split": row.split,
            }
        )
    external = pd.DataFrame(external_rows)
    holdout = external.loc[
        external["external_split"].eq("holdout")
    ].reset_index(drop=True)

    expected = (
        {"fine_tune": 34, "minoxidil": 20, "excluded": 0}
        if args.mode == "final"
        else {"fine_tune": 19, "minoxidil": 5, "excluded": 15}
    )
    if (
        len(fine_tune) != expected["fine_tune"]
        or len(minoxidil) != expected["minoxidil"]
        or len(external) != 64
        or len(holdout) != 50
        or len(excluded_locks) != expected["excluded"]
    ):
        raise AssertionError(
            f"Unexpected counts for mode={args.mode}: "
            f"fine_tune={len(fine_tune)}, minoxidil={len(minoxidil)}, "
            f"external={len(external)}, holdout={len(holdout)}, "
            f"excluded={len(excluded_locks)}"
        )
    validate_unique_pairs(fine_tune, "Fine-tuning table")
    validate_unique_pairs(minoxidil, "Minoxidil table")
    validate_unique_pairs(external, "External 64 table")
    validate_unique_pairs(holdout, "External holdout")
    overlap = set(fine_tune["pair_key"]).intersection(holdout["pair_key"])
    if overlap:
        raise ValueError(
            f"Fine-tuning and holdout tables overlap on {len(overlap)} pairs"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "final":
        outputs = {
            "four_class_finetune_34_physical_pairs.csv": fine_tune,
            "four_class_minoxidil_20_physical_pairs.csv": minoxidil,
            "binary_finetune_34_physical_pairs.csv": binary_view(fine_tune),
            "binary_minoxidil_20_physical_pairs.csv": binary_view(minoxidil),
        }
    else:
        outputs = {
            "four_class_provisional_finetune_19_physical_pairs.csv": fine_tune,
            "four_class_provisional_minoxidil_5_physical_pairs.csv": minoxidil,
            "binary_provisional_finetune_19_physical_pairs.csv": (
                binary_view(fine_tune)
            ),
            "binary_provisional_minoxidil_5_physical_pairs.csv": (
                binary_view(minoxidil)
            ),
        }
    outputs.update(
        {
            "four_class_external_64_physical_pairs.csv": external,
            "four_class_holdout_50_physical_pairs.csv": holdout,
            "binary_external_64_physical_pairs.csv": binary_view(external),
            "binary_holdout_50_physical_pairs.csv": binary_view(holdout),
        }
    )
    hashes = {}
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        hashes[filename] = sha256_file(path)

    manifest = {
        "schema_version": "mcc-gcn-locked-datasets-v2",
        "mode": args.mode,
        "eligible_for_final_reporting": args.mode == "final",
        "inputs": {
            "finetune_lock": {
                "path": args.finetune_lock,
                "sha256": sha256_file(args.finetune_lock),
            },
            "external_split_lock": {
                "path": args.external_split_lock,
                "sha256": sha256_file(args.external_split_lock),
            },
            "approved_minoxidil_pairs": (
                {
                    "path": args.approved_minoxidil_pairs,
                    "sha256": sha256_file(args.approved_minoxidil_pairs),
                }
                if args.approved_minoxidil_pairs
                else None
            ),
        },
        "counts": {
            "fine_tune_physical_pairs": len(fine_tune),
            "minoxidil_physical_pairs": len(minoxidil),
            "external_physical_pairs": len(external),
            "holdout_physical_pairs": len(holdout),
            "fine_tune_holdout_overlap": 0,
            "excluded_unresolved_physical_pairs": len(excluded_locks),
        },
        "excluded_unresolved_lock_ids": excluded_locks,
        "outputs": hashes,
    }
    (output_dir / "locked_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

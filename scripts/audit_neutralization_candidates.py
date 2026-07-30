#!/usr/bin/env python
"""Assess pair-level neutralization candidates without approving reactants."""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.neutralization import assess_standardized_pair
from mcc_gcn.data.quality import canonical_pair_key, read_pair_table
from mcc_gcn.data.standardize import standardize_exported_component

MECHANICALLY_ELIGIBLE_STATUSES = {
    "observed_neutral_pair",
    "balanced_monovalent_candidate",
    "balanced_multivalent_candidate",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Legacy molecular-pair CSV files.")
    parser.add_argument(
        "--component-molblocks",
        help=(
            "Optional gzip-pickled SMILES-to-MolBlock mapping from the "
            "historical CCDC export."
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


def _serialize(value):
    if isinstance(value, tuple):
        return ";".join(str(item) for item in value)
    return value


def _component_columns(prefix, result):
    return {
        f"{prefix}_{key}": _serialize(value)
        for key, value in result.to_dict().items()
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = [read_pair_table(path) for path in args.inputs]
    combined = pd.concat(tables, ignore_index=True)
    component_molblocks = {}
    if args.component_molblocks:
        with gzip.open(args.component_molblocks, "rb") as handle:
            component_molblocks = pickle.load(handle)
        if not isinstance(component_molblocks, dict):
            raise TypeError("--component-molblocks must contain a dictionary")

    @lru_cache(maxsize=None)
    def standardize_smiles(smiles):
        return standardize_exported_component(
            {
                "smiles": smiles,
                "mol_block": component_molblocks.get(smiles),
            }
        )

    rows = []
    for row in tqdm(
        combined.itertuples(index=False),
        total=len(combined),
        desc="Auditing neutralization candidates",
        unit="pair",
    ):
        component_a = standardize_smiles(row.reactant_A)
        component_b = standardize_smiles(row.reactant_B)
        assessment = assess_standardized_pair(component_a, component_b)
        output = {
            "reactant_A": row.reactant_A,
            "reactant_B": row.reactant_B,
            "label_str": row.label_str,
            "label_int": row.label_int,
            "identifier": row.identifier,
            "source_file": row.source_file,
            "source_row": row.source_row,
            "pair_candidate_status": assessment.status,
            "observed_charge_sum": assessment.observed_charge_sum,
            "candidate_charge_sum": assessment.candidate_charge_sum,
            "hydrogen_delta_sum": assessment.hydrogen_delta_sum,
            "proton_transfer_consistent": (
                assessment.proton_transfer_consistent
            ),
            "pair_warnings": _serialize(assessment.warnings),
        }
        output.update(_component_columns("A", component_a))
        output.update(_component_columns("B", component_b))
        rows.append(output)

    audited = pd.DataFrame(rows)
    mechanically_eligible = audited["pair_candidate_status"].isin(
        MECHANICALLY_ELIGIBLE_STATUSES
    )
    audited["candidate_pair_key"] = None
    audited.loc[mechanically_eligible, "candidate_pair_key"] = audited.loc[
        mechanically_eligible
    ].apply(
        lambda row: canonical_pair_key(
            row["A_candidate_smiles"],
            row["B_candidate_smiles"],
        ),
        axis=1,
    )
    eligible_labels = audited.loc[
        mechanically_eligible,
        ["candidate_pair_key", "label_str", "label_int"],
    ].copy()
    eligible_labels["label_key"] = (
        eligible_labels["label_str"]
        + ":"
        + eligible_labels["label_int"].astype(str)
    )
    candidate_labels = eligible_labels.groupby(
        "candidate_pair_key", sort=False
    )["label_key"].nunique()
    conflict_keys = set(candidate_labels[candidate_labels.gt(1)].index)
    audited["candidate_has_label_conflict"] = audited[
        "candidate_pair_key"
    ].isin(conflict_keys)
    audited.to_csv(output_dir / "neutralization_candidate_rows.csv", index=False)

    by_label_status = (
        audited.groupby(
            ["label_str", "label_int", "pair_candidate_status"],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    by_label_status.to_csv(
        output_dir / "neutralization_status_by_label.csv",
        index=False,
    )
    audited.loc[audited["candidate_has_label_conflict"]].to_csv(
        output_dir / "neutralization_candidate_conflicts.csv",
        index=False,
    )

    component_smiles = set(combined["reactant_A"]).union(combined["reactant_B"])
    unique_components = {
        smiles: standardize_smiles(smiles) for smiles in component_smiles
    }
    component_status_counts = {}
    for result in unique_components.values():
        status = result.candidate_status or "parse_error"
        component_status_counts[status] = component_status_counts.get(status, 0) + 1

    eligible_rows = audited.loc[mechanically_eligible]
    summary = {
        "schema_version": "mcc-gcn-neutralization-candidate-audit-v1",
        "inputs": [
            {
                "path": str(Path(path)),
                "sha256": sha256_file(path),
            }
            for path in args.inputs
        ],
        "component_molblocks": (
            {
                "path": str(Path(args.component_molblocks)),
                "sha256": sha256_file(args.component_molblocks),
            }
            if args.component_molblocks
            else None
        ),
        "rows": len(audited),
        "unique_input_components": len(unique_components),
        "mechanically_eligible_rows": len(eligible_rows),
        "mechanically_eligible_unique_unordered_pairs": int(
            eligible_rows["candidate_pair_key"].nunique()
        ),
        "mechanically_eligible_duplicate_rows": int(
            len(eligible_rows) - eligible_rows["candidate_pair_key"].nunique()
        ),
        "candidate_conflicting_unordered_pairs": len(conflict_keys),
        "candidate_conflicting_rows": int(
            audited["candidate_has_label_conflict"].sum()
        ),
        "pair_candidate_status_counts": {
            str(status): int(count)
            for status, count in audited["pair_candidate_status"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "component_candidate_status_counts": dict(
            sorted(component_status_counts.items())
        ),
    }
    (output_dir / "neutralization_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Neutralization candidate audit written to {output_dir}")


if __name__ == "__main__":
    main()

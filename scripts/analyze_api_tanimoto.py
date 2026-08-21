#!/usr/bin/env python3
"""Calculate reviewer-requested API and target/pretraining similarities."""

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

from mcc_gcn.data.similarity import (
    MoleculeReference,
    nearest_pretraining_pairs,
    pairwise_tanimoto,
)


API_REFERENCES = [
    MoleculeReference(
        "Minoxidil",
        "N=C1N=C(N2CCCCC2)C=C(N)N1O",
        "legacy_fine_tuning_api",
    ),
    MoleculeReference(
        "Kopyxil",
        "Nc1ccnc(N)[n+]1[O-]",
        "legacy_fine_tuning_api",
    ),
    MoleculeReference(
        "Kopyrrol",
        "Nc1cc(N2CCCC2)nc(N)[n+]1[O-]",
        "legacy_fine_tuning_api",
    ),
    MoleculeReference(
        "Paracetamol",
        "CC(=O)Nc1ccc(O)cc1",
        "revision_target_api",
    ),
    MoleculeReference(
        "Theophylline",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "revision_target_api",
    ),
    MoleculeReference(
        "Riluzole",
        "Nc1nc2ccc(OC(F)(F)F)cc2s1",
        "revision_target_api",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pairs", required=True)
    parser.add_argument("--pretraining-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument(
        "--no-chirality",
        action="store_true",
        help="Disable chirality in Morgan fingerprints.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: pd.Series) -> dict[str, float | int]:
    values = values.astype(float)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "maximum": float(values.max()),
        "exact_match_fraction": float(values.eq(1.0).mean()),
    }


def grouped_summary(table: pd.DataFrame, column: str) -> dict[str, dict]:
    result = {}
    for value, group in table.groupby(column, sort=True):
        result[str(value)] = {
            "pairs": int(len(group)),
            "api_nearest_component": distribution(
                group["api_nearest_component_tanimoto"]
            ),
            "coformer_nearest_component": distribution(
                group["coformer_nearest_component_tanimoto"]
            ),
            "pair_nearest": distribution(group["pair_nearest_tanimoto"]),
        }
    return result


def pair_rows(table: pd.DataFrame, names: set[str]) -> list[dict]:
    mask = table["molecule_1"].isin(names) & table["molecule_2"].isin(names)
    return [
        {
            "molecule_1": row.molecule_1,
            "molecule_2": row.molecule_2,
            "tanimoto": float(row.tanimoto),
        }
        for row in table.loc[mask].itertuples(index=False)
    ]


def main():
    args = parse_args()
    include_chirality = not args.no_chirality
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(args.target_pairs, keep_default_na=False)
    pretraining = pd.read_csv(args.pretraining_pairs, keep_default_na=False)

    matrix, pairwise = pairwise_tanimoto(
        API_REFERENCES,
        radius=args.radius,
        n_bits=args.n_bits,
        include_chirality=include_chirality,
    )
    nearest = nearest_pretraining_pairs(
        target,
        pretraining,
        radius=args.radius,
        n_bits=args.n_bits,
        include_chirality=include_chirality,
    )
    matrix_path = output_dir / "api_pairwise_tanimoto_matrix.csv"
    pairwise_path = output_dir / "api_pairwise_tanimoto_long.csv"
    nearest_path = output_dir / "target_nearest_pretraining_pairs.csv"
    matrix.to_csv(matrix_path)
    pairwise.to_csv(pairwise_path, index=False)
    nearest.to_csv(nearest_path, index=False)

    legacy_names = {
        reference.name
        for reference in API_REFERENCES
        if reference.group == "legacy_fine_tuning_api"
    }
    revision_names = {
        reference.name
        for reference in API_REFERENCES
        if reference.group == "revision_target_api"
    }
    summary = {
        "schema_version": "mcc-gcn-api-tanimoto-v1",
        "method": {
            "fingerprint": "Morgan/ECFP4",
            "radius": args.radius,
            "n_bits": args.n_bits,
            "include_chirality": include_chirality,
            "metric": "Tanimoto",
            "pair_similarity": (
                "maximum mean component Tanimoto over both assignments"
            ),
        },
        "software": {"rdkit": rdBase.rdkitVersion},
        "inputs": {
            "target_pairs": {
                "path": str(Path(args.target_pairs)),
                "sha256": sha256_file(args.target_pairs),
                "physical_pairs": int(len(target)),
            },
            "pretraining_pairs": {
                "path": str(Path(args.pretraining_pairs)),
                "sha256": sha256_file(args.pretraining_pairs),
                "physical_pairs": int(len(pretraining)),
            },
        },
        "legacy_fine_tuning_api_pairwise": pair_rows(
            pairwise,
            legacy_names,
        ),
        "revision_target_api_pairwise": pair_rows(
            pairwise,
            revision_names,
        ),
        "target_vs_pretraining": {
            "overall": {
                "pairs": int(len(nearest)),
                "api_nearest_component": distribution(
                    nearest["api_nearest_component_tanimoto"]
                ),
                "coformer_nearest_component": distribution(
                    nearest["coformer_nearest_component_tanimoto"]
                ),
                "pair_nearest": distribution(
                    nearest["pair_nearest_tanimoto"]
                ),
            },
            "by_target_api": grouped_summary(nearest, "target_apis"),
            "by_label": grouped_summary(nearest, "label_str"),
        },
        "outputs": {
            matrix_path.name: sha256_file(matrix_path),
            pairwise_path.name: sha256_file(pairwise_path),
            nearest_path.name: sha256_file(nearest_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

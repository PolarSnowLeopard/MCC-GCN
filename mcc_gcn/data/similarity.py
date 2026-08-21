"""Molecular and molecular-pair similarity utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator


@dataclass(frozen=True)
class MoleculeReference:
    name: str
    smiles: str
    group: str


def _fingerprint_generator(radius: int, n_bits: int, include_chirality: bool):
    if radius < 1:
        raise ValueError("Fingerprint radius must be positive")
    if n_bits < 1:
        raise ValueError("Fingerprint size must be positive")
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )


def _fingerprint(smiles: str, generator):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return generator.GetFingerprint(molecule)


def pairwise_tanimoto(
    references: list[MoleculeReference],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return square and long-form Tanimoto tables for named molecules."""
    if not references:
        raise ValueError("At least one molecule reference is required")
    names = [reference.name for reference in references]
    if len(names) != len(set(names)):
        raise ValueError("Molecule reference names must be unique")

    generator = _fingerprint_generator(radius, n_bits, include_chirality)
    fingerprints = {
        reference.name: _fingerprint(reference.smiles, generator)
        for reference in references
    }
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    rows = []
    for left_index, left in enumerate(references):
        for right_index, right in enumerate(references):
            similarity = DataStructs.TanimotoSimilarity(
                fingerprints[left.name],
                fingerprints[right.name],
            )
            matrix.loc[left.name, right.name] = similarity
            if left_index < right_index:
                rows.append(
                    {
                        "molecule_1": left.name,
                        "group_1": left.group,
                        "smiles_1": left.smiles,
                        "molecule_2": right.name,
                        "group_2": right.group,
                        "smiles_2": right.smiles,
                        "tanimoto": similarity,
                    }
                )
    matrix.index.name = "molecule"
    return matrix, pd.DataFrame(rows)


def nearest_pretraining_pairs(
    target_pairs: pd.DataFrame,
    pretraining_pairs: pd.DataFrame,
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> pd.DataFrame:
    """Find nearest pretraining components and whole pairs for target pairs.

    Whole-pair similarity is the maximum mean component similarity over both
    possible component assignments. The target table is expected to place its
    API in ``reactant_A`` and its coformer in ``reactant_B``.
    """
    required_target = {
        "reactant_A",
        "reactant_B",
        "pair_key",
        "target_apis",
        "label_str",
        "label_int",
    }
    required_pretraining = {"reactant_A", "reactant_B", "pair_key"}
    missing_target = required_target.difference(target_pairs.columns)
    missing_pretraining = required_pretraining.difference(
        pretraining_pairs.columns
    )
    if missing_target:
        raise ValueError(
            f"Target pairs are missing columns: {sorted(missing_target)}"
        )
    if missing_pretraining:
        raise ValueError(
            "Pretraining pairs are missing columns: "
            f"{sorted(missing_pretraining)}"
        )
    if target_pairs.empty or pretraining_pairs.empty:
        raise ValueError("Target and pretraining pair tables must be non-empty")
    if target_pairs["pair_key"].duplicated().any():
        raise ValueError("Target pairs must be unique")
    if pretraining_pairs["pair_key"].duplicated().any():
        raise ValueError("Pretraining pairs must be unique")

    target = target_pairs.sort_values("pair_key").reset_index(drop=True)
    pretraining = pretraining_pairs.sort_values("pair_key").reset_index(
        drop=True
    )
    generator = _fingerprint_generator(radius, n_bits, include_chirality)
    all_smiles = sorted(
        set(target["reactant_A"])
        | set(target["reactant_B"])
        | set(pretraining["reactant_A"])
        | set(pretraining["reactant_B"])
    )
    cache = {
        smiles: _fingerprint(smiles, generator) for smiles in all_smiles
    }
    pre_a = [cache[smiles] for smiles in pretraining["reactant_A"]]
    pre_b = [cache[smiles] for smiles in pretraining["reactant_B"]]
    component_smiles = sorted(
        set(pretraining["reactant_A"]) | set(pretraining["reactant_B"])
    )
    component_fingerprints = [cache[smiles] for smiles in component_smiles]

    rows = []
    for row in target.itertuples(index=False):
        api_fp = cache[row.reactant_A]
        coformer_fp = cache[row.reactant_B]
        api_to_a = np.asarray(DataStructs.BulkTanimotoSimilarity(api_fp, pre_a))
        api_to_b = np.asarray(DataStructs.BulkTanimotoSimilarity(api_fp, pre_b))
        coformer_to_a = np.asarray(
            DataStructs.BulkTanimotoSimilarity(coformer_fp, pre_a)
        )
        coformer_to_b = np.asarray(
            DataStructs.BulkTanimotoSimilarity(coformer_fp, pre_b)
        )
        direct = (api_to_a + coformer_to_b) / 2.0
        reverse = (api_to_b + coformer_to_a) / 2.0
        pair_scores = np.maximum(direct, reverse)
        pair_index = int(np.argmax(pair_scores))
        reverse_match = bool(reverse[pair_index] > direct[pair_index])

        api_component_scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                api_fp,
                component_fingerprints,
            )
        )
        coformer_component_scores = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                coformer_fp,
                component_fingerprints,
            )
        )
        api_component_index = int(np.argmax(api_component_scores))
        coformer_component_index = int(np.argmax(coformer_component_scores))
        nearest_pair = pretraining.iloc[pair_index]
        rows.append(
            {
                "pair_key": row.pair_key,
                "target_apis": row.target_apis,
                "label_str": row.label_str,
                "label_int": int(row.label_int),
                "api_smiles": row.reactant_A,
                "coformer_smiles": row.reactant_B,
                "api_nearest_component_tanimoto": float(
                    api_component_scores[api_component_index]
                ),
                "api_nearest_component_smiles": component_smiles[
                    api_component_index
                ],
                "coformer_nearest_component_tanimoto": float(
                    coformer_component_scores[coformer_component_index]
                ),
                "coformer_nearest_component_smiles": component_smiles[
                    coformer_component_index
                ],
                "pair_nearest_tanimoto": float(pair_scores[pair_index]),
                "nearest_pretraining_pair_key": nearest_pair["pair_key"],
                "nearest_pretraining_reactant_A": nearest_pair["reactant_A"],
                "nearest_pretraining_reactant_B": nearest_pair["reactant_B"],
                "nearest_pair_component_assignment": (
                    "reverse" if reverse_match else "direct"
                ),
            }
        )
    return pd.DataFrame(rows)

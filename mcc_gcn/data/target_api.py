"""Curate and split the three-API revision experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.model_selection import StratifiedKFold

from .quality import canonical_pair_key


@dataclass(frozen=True)
class TargetAPI:
    name: str
    cas_number: str
    connectivity_key: str


TARGET_APIS = (
    TargetAPI("Paracetamol", "103-90-2", "RZVAJINKPMORJF"),
    TargetAPI("Theophylline", "58-55-9", "ZFXYFBGIUFBOJW"),
    TargetAPI("Riluzole", "1744-22-5", "FTALBRSUTCGOEG"),
)

FOUR_CLASS_LABELS = {
    "failed": ("negative", 0),
    "negative": ("negative", 0),
    "salt": ("salt", 1),
    "cocrystal": ("cocrystal", 2),
    "hydrate": ("hydrate_or_solvate", 3),
    "solvate": ("hydrate_or_solvate", 3),
    "hydrate_or_solvate": ("hydrate_or_solvate", 3),
}


@dataclass(frozen=True)
class TargetAPIResult:
    evidence: pd.DataFrame
    conflicts: pd.DataFrame
    four_class_pairs: pd.DataFrame
    binary_pairs: pd.DataFrame
    pretrain_exclusions: pd.DataFrame
    four_class_pretrain_pairs: pd.DataFrame
    binary_pretrain_pairs: pd.DataFrame


def molecule_identity(smiles: str) -> tuple[str, str]:
    """Return canonical SMILES and the connectivity block of an InChIKey."""
    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("empty SMILES")
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES: {smiles}")
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
            inchi_key = Chem.MolToInchiKey(molecule)
    except Exception as exc:  # noqa: BLE001 - RDKit errors vary by release
        raise ValueError(f"invalid SMILES: {smiles}") from exc
    if not inchi_key:
        raise ValueError(f"InChIKey generation failed: {smiles}")
    return canonical, inchi_key.split("-", maxsplit=1)[0]


def connectivity_pair_key(key_a: str, key_b: str) -> str:
    """Serialize an order-independent pair of connectivity identities."""
    return json.dumps(sorted((key_a, key_b)), separators=(",", ":"))


def _target_names(
    key_a: str,
    key_b: str,
    targets: tuple[TargetAPI, ...],
) -> tuple[str, ...]:
    return tuple(
        target.name
        for target in targets
        if target.connectivity_key in {key_a, key_b}
    )


def _annotate_components(
    table: pd.DataFrame,
    *,
    targets: tuple[TargetAPI, ...],
    strict: bool = True,
) -> pd.DataFrame:
    result = table.copy().reset_index(drop=True)
    cache: dict[str, tuple[str, str]] = {}
    for side in ("A", "B"):
        canonical_values = []
        connectivity_values = []
        error_values = []
        for smiles in result[f"reactant_{side}"]:
            if smiles not in cache:
                try:
                    cache[smiles] = molecule_identity(smiles)
                except ValueError:
                    if strict:
                        raise
                    cache[smiles] = ("", "")
            canonical, connectivity = cache[smiles]
            canonical_values.append(canonical)
            connectivity_values.append(connectivity)
            error_values.append("" if connectivity else "invalid_smiles")
        result[f"canonical_{side}"] = canonical_values
        result[f"connectivity_{side}"] = connectivity_values
        result[f"component_{side}_error"] = error_values
    result["connectivity_pair_key"] = [
        connectivity_pair_key(key_a, key_b) if key_a and key_b else ""
        for key_a, key_b in zip(
            result["connectivity_A"],
            result["connectivity_B"],
        )
    ]
    result["target_apis"] = [
        ";".join(_target_names(key_a, key_b, targets))
        for key_a, key_b in zip(
            result["connectivity_A"],
            result["connectivity_B"],
        )
    ]
    return result


def _normalize_labels(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    source_labels = result["label_str"].astype(str).str.strip().str.lower()
    unknown = sorted(set(source_labels).difference(FOUR_CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unsupported labels: {unknown}")
    normalized = source_labels.map(FOUR_CLASS_LABELS)
    result["source_label_str"] = source_labels
    result["label_str"] = [value[0] for value in normalized]
    result["label_int"] = [value[1] for value in normalized]
    return result


def _orient_representative(
    row: pd.Series,
    targets: tuple[TargetAPI, ...],
) -> tuple[str, str]:
    target_keys = [target.connectivity_key for target in targets]
    key_a = row["connectivity_A"]
    key_b = row["connectivity_B"]
    if key_a in target_keys and key_b not in target_keys:
        return row["canonical_A"], row["canonical_B"]
    if key_b in target_keys and key_a not in target_keys:
        return row["canonical_B"], row["canonical_A"]
    if key_a in target_keys and key_b in target_keys:
        if target_keys.index(key_a) <= target_keys.index(key_b):
            return row["canonical_A"], row["canonical_B"]
        return row["canonical_B"], row["canonical_A"]
    raise AssertionError("Representative does not contain a target API")


def _collapse_target_pairs(
    evidence: pd.DataFrame,
    targets: tuple[TargetAPI, ...],
) -> pd.DataFrame:
    rows = []
    for pair_key, group in evidence.groupby(
        "connectivity_pair_key",
        sort=True,
    ):
        labels = set(group["label_str"])
        if len(labels) != 1:
            raise AssertionError(f"Unresolved label conflict: {pair_key}")
        representative = group.sort_values(
            [
                "source_kind",
                "identifier",
                "source_row",
                "canonical_A",
                "canonical_B",
            ],
            kind="stable",
        ).iloc[0]
        reactant_a, reactant_b = _orient_representative(
            representative,
            targets,
        )
        target_names = tuple(
            target.name
            for target in targets
            if target.connectivity_key
            in {
                representative["connectivity_A"],
                representative["connectivity_B"],
            }
        )
        identifiers = sorted(
            value
            for value in set(group["identifier"].astype(str))
            if value
        )
        source_rows = sorted(
            f"{row.source_kind}:{row.source_row}"
            for row in group.itertuples(index=False)
        )
        label_str = representative["label_str"]
        label_int = int(representative["label_int"])
        rows.append(
            {
                "reactant_A": reactant_a,
                "reactant_B": reactant_b,
                "label_str": label_str,
                "label_int": label_int,
                "identifier": ";".join(identifiers),
                "pair_key": pair_key,
                "target_apis": ";".join(target_names),
                "evidence_count": len(group),
                "source_kind": ";".join(
                    sorted(set(group["source_kind"]))
                ),
                "source_rows": ";".join(source_rows),
                "representative_pair_key": canonical_pair_key(
                    reactant_a,
                    reactant_b,
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result["pair_key"].nunique() != len(result):
        raise AssertionError("Target table contains duplicate physical pairs")
    return result


def curate_target_api_experiment(
    csd_pairs: pd.DataFrame,
    negative_pairs: pd.DataFrame,
    pretrain_pairs: pd.DataFrame,
    *,
    targets: tuple[TargetAPI, ...] = TARGET_APIS,
) -> TargetAPIResult:
    """Build the target benchmark and remove all target APIs from pretraining."""
    required = {
        "reactant_A",
        "reactant_B",
        "label_str",
        "label_int",
        "identifier",
    }
    for name, table in {
        "csd_pairs": csd_pairs,
        "negative_pairs": negative_pairs,
        "pretrain_pairs": pretrain_pairs,
    }.items():
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    evidence_parts = []
    for source_kind, source in (
        ("csd_observation", csd_pairs),
        ("failed_cocrystallization", negative_pairs),
    ):
        part = source.copy().reset_index(drop=True)
        if "source_row" not in part:
            part["source_row"] = np.arange(1, len(part) + 1)
        part["source_kind"] = source_kind
        evidence_parts.append(part)
    evidence = pd.concat(evidence_parts, ignore_index=True)
    evidence = _normalize_labels(evidence)
    evidence = _annotate_components(evidence, targets=targets, strict=False)
    evidence = evidence.loc[evidence["target_apis"].ne("")].copy()

    label_counts = evidence.groupby("connectivity_pair_key")[
        "label_str"
    ].nunique()
    conflict_keys = set(label_counts[label_counts.gt(1)].index)
    evidence["curation_status"] = np.where(
        evidence["connectivity_pair_key"].isin(conflict_keys),
        "quarantine_label_conflict",
        "candidate_target_benchmark",
    )
    conflicts = evidence.loc[
        evidence["connectivity_pair_key"].isin(conflict_keys)
    ].copy()
    accepted = evidence.loc[
        ~evidence["connectivity_pair_key"].isin(conflict_keys)
    ]
    four_class = _collapse_target_pairs(accepted, targets)
    binary = four_class.copy()
    binary["label_str"] = np.where(
        binary["label_int"].eq(0),
        "negative",
        "positive",
    )
    binary["label_int"] = np.where(binary["label_int"].eq(0), 0, 1)

    annotated_pretrain = _annotate_components(
        pretrain_pairs,
        targets=targets,
        strict=False,
    )
    exclusion_mask = annotated_pretrain["target_apis"].ne("")
    exclusions = annotated_pretrain.loc[exclusion_mask].copy()
    four_class_pretrain = pretrain_pairs.loc[~exclusion_mask].copy()
    binary_pretrain = four_class_pretrain.copy()
    binary_pretrain["label_str"] = np.where(
        binary_pretrain["label_int"].eq(0),
        "negative",
        "positive",
    )
    binary_pretrain["label_int"] = np.where(
        binary_pretrain["label_int"].eq(0),
        0,
        1,
    )
    if _annotate_components(
        four_class_pretrain,
        targets=targets,
        strict=False,
    )["target_apis"].ne("").any():
        raise AssertionError("Target API remained in pretraining data")

    return TargetAPIResult(
        evidence=evidence.reset_index(drop=True),
        conflicts=conflicts.reset_index(drop=True),
        four_class_pairs=four_class.reset_index(drop=True),
        binary_pairs=binary.reset_index(drop=True),
        pretrain_exclusions=exclusions.reset_index(drop=True),
        four_class_pretrain_pairs=four_class_pretrain.reset_index(drop=True),
        binary_pretrain_pairs=binary_pretrain.reset_index(drop=True),
    )


def stratified_target_folds(
    table: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each physical pair to exactly one stratified outer test fold."""
    required = {"pair_key", "label_int", "target_apis"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    if table["pair_key"].nunique() != len(table):
        raise ValueError("Target table must contain one row per physical pair")
    class_counts = table["label_int"].value_counts()
    if class_counts.min() < n_splits:
        raise ValueError(
            "Every class needs at least n_splits physical pairs"
        )

    ordered = table.sort_values("pair_key").reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    test_fold = np.full(len(ordered), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(
        splitter.split(ordered, ordered["label_int"])
    ):
        test_fold[test_indices] = fold
    if np.any(test_fold < 0):
        raise AssertionError("At least one target pair has no test fold")
    result = ordered[["pair_key", "label_str", "label_int", "target_apis"]].copy()
    result["test_fold"] = test_fold
    result["n_splits"] = n_splits
    result["seed"] = seed
    return result.sort_values(["test_fold", "label_int", "pair_key"]).reset_index(
        drop=True
    )

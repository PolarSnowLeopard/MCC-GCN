"""Build task tables from preserved evidence without silent relabeling."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from .quality import (
    DEFAULT_ALLOWED_ELEMENTS,
    DEFAULT_ALLOWED_HYBRIDIZATIONS,
    audit_pair_table,
    audit_smiles,
)

ACCEPTED_CSD_STATUSES = {
    "observed_neutral_pair",
    "balanced_monovalent_candidate",
    "balanced_multivalent_candidate",
}

FOUR_CLASS_LABELS = {
    "failed": ("failed", 0),
    "negative": ("failed", 0),
    "salt": ("salt", 1),
    "cocrystal": ("cocrystal", 2),
    "hydrate": ("hydrate_or_solvate", 3),
    "solvate": ("hydrate_or_solvate", 3),
}


@dataclass(frozen=True)
class CurationResult:
    evidence: pd.DataFrame
    conflicts: pd.DataFrame
    four_class_pairs: pd.DataFrame
    binary_pairs: pd.DataFrame


def _as_boolean(series):
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _elements_supported(row):
    elements = set()
    for side in ("A", "B"):
        value = str(row[f"{side}_elements"])
        elements.update(item for item in value.split(";") if item)
    return elements.issubset(DEFAULT_ALLOWED_ELEMENTS)


def _hybridizations_supported(smiles, cache):
    if smiles not in cache:
        cache[smiles] = audit_smiles(smiles)
    audit = cache[smiles]
    return (
        audit.error is None
        and set(audit.hybridizations).issubset(
            DEFAULT_ALLOWED_HYBRIDIZATIONS
        )
    )


def _append_reason(table, mask, reason):
    table.loc[mask, "curation_reasons"] = table.loc[
        mask,
        "curation_reasons",
    ].map(lambda values: [*values, reason])


def _status_from_reasons(reasons):
    if not reasons:
        return "candidate_training"
    if "locked_finetune_identifier" in reasons:
        return "locked_finetune_identifier"
    if "locked_external_or_finetune_pair" in reasons:
        return "locked_external_or_finetune_pair"
    if any("conflict" in reason for reason in reasons):
        return "quarantine_label_conflict"
    return "quarantine_quality"


def _collapse_task_pairs(evidence):
    rows = []
    for pair_key, group in evidence.groupby("pair_key", sort=True):
        source_labels = set(group["source_label_str"])
        if len(source_labels) != 1:
            raise AssertionError(
                f"Unresolved source-label conflict for {pair_key}"
            )
        source_label = next(iter(source_labels))
        label_str, label_int = FOUR_CLASS_LABELS[source_label]
        smiles_a, smiles_b = json.loads(pair_key)
        identifiers = sorted(
            {
                value
                for value in group["identifier"].astype(str)
                if value
            }
        )
        rows.append(
            {
                "reactant_A": smiles_a,
                "reactant_B": smiles_b,
                "label_str": label_str,
                "label_int": label_int,
                "identifier": ";".join(identifiers),
                "pair_key": pair_key,
                "evidence_count": len(group),
                "source_kind": ";".join(
                    sorted(set(group["source_kind"]))
                ),
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "reactant_A",
            "reactant_B",
            "label_str",
            "label_int",
            "identifier",
            "pair_key",
            "evidence_count",
            "source_kind",
        ],
    )
    if not result.empty and result["pair_key"].nunique() != len(result):
        raise AssertionError("Task table contains duplicate pair keys")
    return result


def curate_pretraining_pairs(
    csd_audit,
    negative_audit,
    *,
    locked_pair_keys=(),
    locked_identifiers=(),
) -> CurationResult:
    """Curate binary and four-class physical-pair tables."""
    locked_pair_keys = set(locked_pair_keys)
    locked_identifiers = set(locked_identifiers)

    csd = csd_audit.copy()
    csd["source_kind"] = "csd_observation"
    csd["source_label_str"] = csd["label_str"].str.lower()
    csd["pair_key"] = csd["candidate_pair_key"].fillna("")
    csd["model_reactant_A"] = csd["A_candidate_smiles"].fillna("")
    csd["model_reactant_B"] = csd["B_candidate_smiles"].fillna("")
    mechanically_eligible = csd["pair_candidate_status"].isin(
        ACCEPTED_CSD_STATUSES
    )
    csd["curation_reasons"] = [
        (
            []
            if eligible
            else [f"pair_status:{status}"]
        )
        for eligible, status in zip(
            mechanically_eligible,
            csd["pair_candidate_status"],
        )
    ]
    label_conflict = _as_boolean(csd["candidate_has_label_conflict"])
    _append_reason(
        csd,
        mechanically_eligible & label_conflict,
        "candidate_label_conflict",
    )
    identical = (
        csd["A_candidate_inchi_key"].fillna("")
        == csd["B_candidate_inchi_key"].fillna("")
    ) & csd["A_candidate_inchi_key"].fillna("").ne("")
    _append_reason(
        csd,
        mechanically_eligible & identical,
        "identical_components",
    )
    supported = csd.apply(_elements_supported, axis=1)
    _append_reason(
        csd,
        mechanically_eligible & ~supported,
        "unsupported_elements",
    )
    molecule_audits = {}
    supported_hybridizations = [
        _hybridizations_supported(smiles_a, molecule_audits)
        and _hybridizations_supported(smiles_b, molecule_audits)
        for smiles_a, smiles_b in zip(
            csd["model_reactant_A"],
            csd["model_reactant_B"],
        )
    ]
    _append_reason(
        csd,
        mechanically_eligible & ~pd.Series(
            supported_hybridizations,
            index=csd.index,
        ),
        "unsupported_hybridization",
    )
    _append_reason(
        csd,
        csd["identifier"].isin(locked_identifiers),
        "locked_finetune_identifier",
    )
    _append_reason(
        csd,
        csd["pair_key"].isin(locked_pair_keys),
        "locked_external_or_finetune_pair",
    )
    csd["curation_status"] = csd["curation_reasons"].map(
        _status_from_reasons
    )

    negative = audit_pair_table(negative_audit).copy()
    negative["source_kind"] = "failed_cocrystallization"
    negative["source_label_str"] = "failed"
    negative["model_reactant_A"] = negative["A_canonical_smiles"].fillna("")
    negative["model_reactant_B"] = negative["B_canonical_smiles"].fillna("")
    negative["curation_reasons"] = [
        [] if clean else ["invalid_negative"]
        for clean in negative["is_clean_candidate"]
    ]
    _append_reason(
        negative,
        negative["pair_key"].isin(locked_pair_keys),
        "locked_external_or_finetune_pair",
    )
    negative["curation_status"] = negative["curation_reasons"].map(
        _status_from_reasons
    )

    csd_training_keys = set(
        csd.loc[
            csd["curation_status"].eq("candidate_training"),
            "pair_key",
        ]
    )
    negative_training_keys = set(
        negative.loc[
            negative["curation_status"].eq("candidate_training"),
            "pair_key",
        ]
    )
    cross_source_conflicts = csd_training_keys.intersection(
        negative_training_keys
    )
    _append_reason(
        csd,
        csd["pair_key"].isin(cross_source_conflicts),
        "cross_source_label_conflict",
    )
    _append_reason(
        negative,
        negative["pair_key"].isin(cross_source_conflicts),
        "cross_source_label_conflict",
    )
    csd["curation_status"] = csd["curation_reasons"].map(
        _status_from_reasons
    )
    negative["curation_status"] = negative["curation_reasons"].map(
        _status_from_reasons
    )
    csd["curation_reasons"] = csd["curation_reasons"].map(
        lambda values: ";".join(values)
    )
    negative["curation_reasons"] = negative["curation_reasons"].map(
        lambda values: ";".join(values)
    )

    evidence_columns = [
        "source_kind",
        "source_file",
        "source_row",
        "identifier",
        "source_label_str",
        "label_int",
        "reactant_A",
        "reactant_B",
        "model_reactant_A",
        "model_reactant_B",
        "pair_key",
        "curation_status",
        "curation_reasons",
    ]
    evidence = pd.concat(
        [
            csd[evidence_columns],
            negative[evidence_columns],
        ],
        ignore_index=True,
    )
    conflicts = evidence.loc[
        evidence["curation_reasons"].str.contains("conflict")
    ].copy()
    training_evidence = evidence.loc[
        evidence["curation_status"].eq("candidate_training")
    ]
    four_class = _collapse_task_pairs(training_evidence)
    binary = four_class.copy()
    binary["label_str"] = binary["label_int"].map(
        lambda value: "negative" if value == 0 else "positive"
    )
    binary["label_int"] = binary["label_int"].map(
        lambda value: 0 if value == 0 else 1
    )
    return CurationResult(
        evidence=evidence,
        conflicts=conflicts,
        four_class_pairs=four_class,
        binary_pairs=binary,
    )

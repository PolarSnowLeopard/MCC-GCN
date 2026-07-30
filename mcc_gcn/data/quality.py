"""Deterministic quality checks for molecular-pair classification tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase

PAIR_COLUMNS = [
    "reactant_A",
    "reactant_B",
    "label_str",
    "label_int",
    "identifier",
]

REQUIRED_COLUMNS = {
    "reactant_A",
    "reactant_B",
    "label_str",
    "label_int",
}

DEFAULT_ALLOWED_ELEMENTS = frozenset(
    {"C", "H", "O", "N", "P", "S", "F", "Cl", "Br", "I", "Si"}
)

# Historical tables used solvate=4 before hydrate/solvate were merged, and
# solvate=3 afterwards. Both encodings are valid inputs to the audit.
VALID_LABEL_ENCODINGS = {
    "failed": {0},
    "negative": {0},
    "neg": {0},
    "salt": {1},
    "positive": {1},
    "pos": {1},
    "cocrystal": {2},
    "hydrate": {3},
    "solvate": {3, 4},
}


@dataclass(frozen=True)
class MoleculeAudit:
    canonical_smiles: str | None
    formal_charge: int | None
    radical_electrons: int | None
    fragment_count: int | None
    heavy_atom_count: int | None
    elements: tuple[str, ...]
    error: str | None


def read_pair_table(path: str | Path) -> pd.DataFrame:
    """Read a historical pair table with or without a CSV header."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError(f"Empty pair table: {path}")

    first_row = {str(value).strip() for value in raw.iloc[0].tolist()}
    has_header = REQUIRED_COLUMNS.issubset(first_row)
    if has_header:
        header = [str(value).strip() for value in raw.iloc[0].tolist()]
        raw = raw.iloc[1:].reset_index(drop=True)
        raw.columns = header
    elif raw.shape[1] in (4, 5):
        raw.columns = PAIR_COLUMNS[: raw.shape[1]]
    else:
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected four or five"
        )

    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "identifier" not in raw:
        raw["identifier"] = ""

    result = raw[PAIR_COLUMNS].copy()
    for column in ("reactant_A", "reactant_B", "label_str", "identifier"):
        result[column] = result[column].astype(str).str.strip()
    result["label_str"] = result["label_str"].str.lower()
    result["label_int"] = pd.to_numeric(
        result["label_int"], errors="coerce"
    ).astype("Int64")
    result["source_file"] = str(path)
    first_data_row = 2 if has_header else 1
    result["source_row"] = range(first_data_row, first_data_row + len(result))
    return result


def audit_smiles(smiles: str) -> MoleculeAudit:
    """Parse and sanitize one SMILES without changing its charge state."""
    smiles = str(smiles).strip()
    if not smiles:
        return MoleculeAudit(None, None, None, None, None, (), "empty_smiles")

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return MoleculeAudit(
                    None, None, None, None, None, (), "parse_error"
                )
            Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit exception types vary by build
        return MoleculeAudit(
            None,
            None,
            None,
            None,
            None,
            (),
            f"sanitization_error:{type(exc).__name__}",
        )

    return MoleculeAudit(
        canonical_smiles=Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        ),
        formal_charge=sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        radical_electrons=sum(
            atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()
        ),
        fragment_count=len(Chem.GetMolFrags(mol)),
        heavy_atom_count=mol.GetNumHeavyAtoms(),
        elements=tuple(sorted({atom.GetSymbol() for atom in mol.GetAtoms()})),
        error=None,
    )


def canonical_pair_key(smiles_a: str, smiles_b: str) -> str:
    """Return a stable, order-independent serialized molecular-pair key."""
    return json.dumps(sorted((smiles_a, smiles_b)), separators=(",", ":"))


def _label_encoding_is_valid(label_str: str, label_int: object) -> bool:
    if pd.isna(label_int):
        return False
    allowed = VALID_LABEL_ENCODINGS.get(label_str)
    return allowed is not None and int(label_int) in allowed


def audit_pair_table(
    table: pd.DataFrame,
    *,
    allowed_elements: Iterable[str] = DEFAULT_ALLOWED_ELEMENTS,
) -> pd.DataFrame:
    """Annotate every row with structural, label, and pair-level issues."""
    missing = set(PAIR_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"Pair table is missing columns: {sorted(missing)}")

    allowed_elements = frozenset(allowed_elements)
    cache: dict[str, MoleculeAudit] = {}
    result = table.copy().reset_index(drop=True)

    for side in ("A", "B"):
        audits = []
        for smiles in result[f"reactant_{side}"]:
            if smiles not in cache:
                cache[smiles] = audit_smiles(smiles)
            audits.append(cache[smiles])
        audit_dicts = [asdict(item) for item in audits]
        for field in MoleculeAudit.__dataclass_fields__:
            result[f"{side}_{field}"] = [item[field] for item in audit_dicts]

    pair_keys: list[str | None] = []
    directed_keys: list[str | None] = []
    issue_lists: list[list[str]] = []
    for row in result.itertuples(index=False):
        issues = []
        if row.A_error:
            issues.append(f"reactant_A_{row.A_error}")
        if row.B_error:
            issues.append(f"reactant_B_{row.B_error}")

        valid_structure = row.A_error is None and row.B_error is None
        if valid_structure:
            if row.A_fragment_count != 1:
                issues.append("reactant_A_disconnected")
            if row.B_fragment_count != 1:
                issues.append("reactant_B_disconnected")
            if row.A_radical_electrons:
                issues.append("reactant_A_radical")
            if row.B_radical_electrons:
                issues.append("reactant_B_radical")
            unsupported = (
                set(row.A_elements) | set(row.B_elements)
            ).difference(allowed_elements)
            if unsupported:
                issues.append(
                    "unsupported_elements:" + ";".join(sorted(unsupported))
                )
            if row.A_canonical_smiles == row.B_canonical_smiles:
                issues.append("identical_reactants")
            pair_keys.append(
                canonical_pair_key(
                    row.A_canonical_smiles, row.B_canonical_smiles
                )
            )
            directed_keys.append(
                json.dumps(
                    [row.A_canonical_smiles, row.B_canonical_smiles],
                    separators=(",", ":"),
                )
            )
        else:
            pair_keys.append(None)
            directed_keys.append(None)

        if not _label_encoding_is_valid(row.label_str, row.label_int):
            issues.append("invalid_label_encoding")
        issue_lists.append(issues)

    result["pair_key"] = pair_keys
    result["directed_key"] = directed_keys
    result["issue_codes"] = [";".join(items) for items in issue_lists]
    result["has_row_issue"] = result["issue_codes"].ne("")

    structurally_eligible = result["pair_key"].notna() & ~result["has_row_issue"]
    eligible = result[structurally_eligible]
    conflict_keys = set()
    for pair_key, group in eligible.groupby("pair_key", sort=False):
        labels = {
            (row.label_str, int(row.label_int))
            for row in group.itertuples(index=False)
        }
        if len(labels) > 1:
            conflict_keys.add(pair_key)

    result["has_label_conflict"] = result["pair_key"].isin(conflict_keys)
    conflict_mask = result["has_label_conflict"]
    result.loc[conflict_mask, "issue_codes"] = result.loc[
        conflict_mask, "issue_codes"
    ].map(lambda value: f"{value};label_conflict".strip(";"))
    result["is_clean_candidate"] = (
        result["pair_key"].notna()
        & ~result["has_row_issue"]
        & ~result["has_label_conflict"]
    )
    return result


def summarize_audit(audited: pd.DataFrame) -> dict:
    """Build a JSON-serializable summary for an audited table."""
    valid_pairs = audited[audited["pair_key"].notna()]
    candidate_pairs = audited[audited["is_clean_candidate"]]
    label_counts = (
        audited.groupby(["label_str", "label_int"], dropna=False)
        .size()
        .sort_index()
    )
    return {
        "rows": len(audited),
        "label_counts": {
            f"{label}:{label_int}": int(count)
            for (label, label_int), count in label_counts.items()
        },
        "invalid_structure_rows": int(audited["pair_key"].isna().sum()),
        "rows_with_any_issue": int(audited["issue_codes"].ne("").sum()),
        "rows_with_radicals": int(
            (
                audited["A_radical_electrons"].fillna(0).gt(0)
                | audited["B_radical_electrons"].fillna(0).gt(0)
            ).sum()
        ),
        "rows_with_nonzero_formal_charge": int(
            (
                audited["A_formal_charge"].fillna(0).ne(0)
                | audited["B_formal_charge"].fillna(0).ne(0)
            ).sum()
        ),
        "unique_unordered_pairs": int(valid_pairs["pair_key"].nunique()),
        "duplicate_rows_by_unordered_pair": int(
            len(valid_pairs) - valid_pairs["pair_key"].nunique()
        ),
        "conflicting_unordered_pairs": int(
            audited.loc[audited["has_label_conflict"], "pair_key"].nunique()
        ),
        "clean_candidate_rows": len(candidate_pairs),
        "clean_candidate_unordered_pairs": int(
            candidate_pairs["pair_key"].nunique()
        ),
    }


def build_clean_pair_table(audited: pd.DataFrame) -> pd.DataFrame:
    """Collapse clean evidence to one deterministic row per unordered pair."""
    rows = []
    clean = audited[audited["is_clean_candidate"]]
    for pair_key, group in clean.groupby("pair_key", sort=True):
        first = group.iloc[0]
        smiles_a, smiles_b = json.loads(pair_key)
        identifiers = sorted(
            {value for value in group["identifier"].astype(str) if value}
        )
        source_files = sorted(set(group["source_file"].astype(str)))
        source_rows = sorted(
            f"{row.source_file}:{row.source_row}"
            for row in group.itertuples(index=False)
        )
        rows.append(
            {
                "reactant_A": smiles_a,
                "reactant_B": smiles_b,
                "label_str": first["label_str"],
                "label_int": int(first["label_int"]),
                "identifiers": ";".join(identifiers),
                "evidence_count": len(group),
                "source_files": ";".join(source_files),
                "source_rows": ";".join(source_rows),
                "pair_key": pair_key,
            }
        )
    return pd.DataFrame(rows)


def build_conflict_table(audited: pd.DataFrame) -> pd.DataFrame:
    """Summarize pair-level label conflicts for manual review."""
    rows = []
    conflicts = audited[audited["has_label_conflict"]]
    for pair_key, group in conflicts.groupby("pair_key", sort=True):
        smiles_a, smiles_b = json.loads(pair_key)
        labels = sorted(
            {
                f"{row.label_str}:{int(row.label_int)}"
                for row in group.itertuples(index=False)
            }
        )
        identifiers = sorted(
            {value for value in group["identifier"].astype(str) if value}
        )
        rows.append(
            {
                "reactant_A": smiles_a,
                "reactant_B": smiles_b,
                "labels": ";".join(labels),
                "identifiers": ";".join(identifiers),
                "evidence_count": len(group),
                "pair_key": pair_key,
            }
        )
    return pd.DataFrame(rows)


def build_source_overlap_table(audited: pd.DataFrame) -> pd.DataFrame:
    """Calculate unordered-pair overlap between every pair of source files."""
    source_keys = {
        source: set(group["pair_key"].dropna())
        for source, group in audited.groupby("source_file", sort=True)
    }
    rows = []
    for source_a, source_b in combinations(sorted(source_keys), 2):
        keys_a = source_keys[source_a]
        keys_b = source_keys[source_b]
        overlap = keys_a.intersection(keys_b)
        rows.append(
            {
                "source_A": source_a,
                "source_B": source_b,
                "pairs_A": len(keys_a),
                "pairs_B": len(keys_b),
                "overlapping_pairs": len(overlap),
                "overlap_fraction_A": (
                    len(overlap) / len(keys_a) if keys_a else 0.0
                ),
                "overlap_fraction_B": (
                    len(overlap) / len(keys_b) if keys_b else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)

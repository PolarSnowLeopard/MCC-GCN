"""CCDC-free component standardization with explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class ComponentStandardization:
    parse_source: str | None
    observed_smiles: str | None
    observed_inchi_key: str | None
    observed_formal_charge: int | None
    parent_smiles: str | None
    parent_inchi_key: str | None
    parent_formal_charge: int | None
    parent_changed: bool | None
    fragment_count: int | None
    heavy_atom_count: int | None
    radical_electrons: int | None
    elements: tuple[str, ...]
    warnings: tuple[str, ...]
    error: str | None

    def to_dict(self):
        return asdict(self)


def _parse_mol_block(value):
    mol = Chem.MolFromMolBlock(
        value,
        sanitize=False,
        removeHs=False,
        strictParsing=True,
    )
    if mol is None:
        raise ValueError("RDKit returned no molecule")
    Chem.SanitizeMol(mol)
    return mol


def _parse_inchi(value):
    mol = Chem.MolFromInchi(value, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("RDKit returned no molecule")
    Chem.SanitizeMol(mol)
    return mol


def _parse_smiles(value):
    mol = Chem.MolFromSmiles(value, sanitize=False)
    if mol is None:
        raise ValueError("RDKit returned no molecule")
    Chem.SanitizeMol(mol)
    return mol


def _parse_component(component):
    candidates = (
        ("mol_block", component.get("mol_block"), _parse_mol_block),
        ("inchi", component.get("inchi"), _parse_inchi),
        ("smiles", component.get("smiles"), _parse_smiles),
    )
    failures = []
    for source, value, parser in candidates:
        if not value:
            continue
        try:
            with rdBase.BlockLogs():
                return parser(value), source, failures
        except Exception as exc:  # noqa: BLE001 - try the next representation
            failures.append(f"{source}:{type(exc).__name__}")
    raise ValueError(
        "no valid molecular representation"
        + (f" ({', '.join(failures)})" if failures else "")
    )


def _inchi_key(mol):
    value = Chem.MolToInchiKey(mol)
    return value or None


def standardize_exported_component(component) -> ComponentStandardization:
    """Create a neutral-parent identity without overwriting observed chemistry."""
    warnings = []
    try:
        mol, parse_source, parse_failures = _parse_component(component)
        warnings.extend(f"fallback_after_{failure}" for failure in parse_failures)

        with rdBase.BlockLogs():
            observed = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
            observed_smiles = Chem.MolToSmiles(
                observed, canonical=True, isomericSmiles=True
            )
            observed_key = _inchi_key(observed)
            observed_charge = Chem.GetFormalCharge(observed)
            fragment_count = len(Chem.GetMolFrags(observed))
            radical_electrons = sum(
                atom.GetNumRadicalElectrons() for atom in observed.GetAtoms()
            )
            elements = tuple(
                sorted({atom.GetSymbol() for atom in observed.GetAtoms()})
            )

            cleaned = rdMolStandardize.Cleanup(Chem.Mol(observed))
            parent = rdMolStandardize.ChargeParent(cleaned)
            Chem.SanitizeMol(parent)
            parent_smiles = Chem.MolToSmiles(
                parent, canonical=True, isomericSmiles=True
            )
            parent_key = _inchi_key(parent)
            parent_charge = Chem.GetFormalCharge(parent)

        if fragment_count != 1:
            warnings.append("observed_component_is_disconnected")
        if radical_electrons:
            warnings.append("observed_component_has_radicals")
        if parent_charge:
            warnings.append("charge_parent_remains_charged")
        if parent.GetNumHeavyAtoms() != observed.GetNumHeavyAtoms():
            warnings.append("charge_parent_changed_heavy_atom_count")

        return ComponentStandardization(
            parse_source=parse_source,
            observed_smiles=observed_smiles,
            observed_inchi_key=observed_key,
            observed_formal_charge=observed_charge,
            parent_smiles=parent_smiles,
            parent_inchi_key=parent_key,
            parent_formal_charge=parent_charge,
            parent_changed=observed_smiles != parent_smiles,
            fragment_count=fragment_count,
            heavy_atom_count=observed.GetNumHeavyAtoms(),
            radical_electrons=radical_electrons,
            elements=elements,
            warnings=tuple(warnings),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - returned as structured failure
        return ComponentStandardization(
            parse_source=None,
            observed_smiles=None,
            observed_inchi_key=None,
            observed_formal_charge=None,
            parent_smiles=None,
            parent_inchi_key=None,
            parent_formal_charge=None,
            parent_changed=None,
            fragment_count=None,
            heavy_atom_count=None,
            radical_electrons=None,
            elements=(),
            warnings=tuple(warnings),
            error=f"{type(exc).__name__}:{exc}",
        )

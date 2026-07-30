"""CCDC-free component standardization with explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from .proton_transfer import enumerate_local_neutralization_candidates


@dataclass(frozen=True)
class ComponentStandardization:
    parse_source: str | None
    observed_smiles: str | None
    observed_inchi_key: str | None
    observed_formal_charge: int | None
    observed_charged_atom_count: int | None
    observed_hydrogen_count: int | None
    candidate_smiles: str | None
    candidate_inchi_key: str | None
    candidate_formal_charge: int | None
    candidate_charged_atom_count: int | None
    candidate_hydrogen_count: int | None
    hydrogen_delta: int | None
    candidate_changed: bool | None
    candidate_method: str | None
    candidate_status: str | None
    local_candidate_status: str | None
    local_candidate_smiles: tuple[str, ...]
    charge_parent_matches_local_candidate: bool | None
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


def _component_representations(component):
    return (
        ("mol_block", component.get("mol_block"), _parse_mol_block),
        ("inchi", component.get("inchi"), _parse_inchi),
        ("smiles", component.get("smiles"), _parse_smiles),
    )


def _inchi_key(mol):
    value = Chem.MolToInchiKey(mol)
    return value or None


def _standardize_parsed_component(
    mol,
    parse_source,
    parse_failures,
) -> ComponentStandardization:
    warnings = [f"fallback_after_{failure}" for failure in parse_failures]
    with rdBase.BlockLogs():
        observed = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
        observed_smiles = Chem.MolToSmiles(
            observed, canonical=True, isomericSmiles=True
        )
        observed_key = _inchi_key(observed)
        observed_charge = Chem.GetFormalCharge(observed)
        observed_charged_atoms = sum(
            atom.GetFormalCharge() != 0 for atom in observed.GetAtoms()
        )
        observed_hydrogens = sum(
            atom.GetTotalNumHs() for atom in observed.GetAtoms()
        )
        fragment_count = len(Chem.GetMolFrags(observed))
        radical_electrons = sum(
            atom.GetNumRadicalElectrons() for atom in observed.GetAtoms()
        )
        elements = tuple(
            sorted({atom.GetSymbol() for atom in observed.GetAtoms()})
        )

        cleaned = rdMolStandardize.Cleanup(Chem.Mol(observed))
        candidate_method = "rdkit_cleanup"
        if observed_charge:
            candidate = rdMolStandardize.ChargeParent(cleaned)
            candidate_method = "rdkit_cleanup_charge_parent"
        else:
            candidate = cleaned
        Chem.SanitizeMol(candidate)
        candidate_smiles = Chem.MolToSmiles(
            candidate, canonical=True, isomericSmiles=True
        )
        candidate_key = _inchi_key(candidate)
        candidate_charge = Chem.GetFormalCharge(candidate)
        candidate_charged_atoms = sum(
            atom.GetFormalCharge() != 0 for atom in candidate.GetAtoms()
        )
        candidate_hydrogens = sum(
            atom.GetTotalNumHs() for atom in candidate.GetAtoms()
        )
        local_candidates = enumerate_local_neutralization_candidates(observed)

    if fragment_count != 1:
        warnings.append("observed_component_is_disconnected")
    if radical_electrons:
        warnings.append("observed_component_has_radicals")
    if candidate_charge:
        warnings.append("candidate_remains_charged")
    if observed_charged_atoms and observed_charge == 0:
        warnings.append("observed_component_has_internal_charges")
    heavy_atoms_changed = candidate.GetNumHeavyAtoms() != observed.GetNumHeavyAtoms()
    if heavy_atoms_changed:
        warnings.append("candidate_changed_heavy_atom_count")

    if fragment_count != 1 or radical_electrons or heavy_atoms_changed:
        candidate_status = "candidate_rejected"
    elif observed_charge == 0:
        candidate_status = "observed_neutral"
    elif candidate_charge == 0:
        candidate_status = "neutralization_candidate_only"
        warnings.append("candidate_requires_pair_level_validation")
    else:
        candidate_status = "unresolved_charged"

    if local_candidates.status == "ambiguous_candidates":
        warnings.append("local_neutralization_site_ambiguous")
    elif local_candidates.status in {
        "configuration_limit_exceeded",
        "no_valid_candidate",
    }:
        warnings.append("no_atom_local_neutralization_candidate")
    if observed_charge:
        charge_parent_matches_local_candidate = (
            candidate_smiles in local_candidates.candidate_smiles
        )
        if not charge_parent_matches_local_candidate:
            warnings.append("charge_parent_not_in_atom_local_candidates")
    else:
        charge_parent_matches_local_candidate = None

    return ComponentStandardization(
        parse_source=parse_source,
        observed_smiles=observed_smiles,
        observed_inchi_key=observed_key,
        observed_formal_charge=observed_charge,
        observed_charged_atom_count=observed_charged_atoms,
        observed_hydrogen_count=observed_hydrogens,
        candidate_smiles=candidate_smiles,
        candidate_inchi_key=candidate_key,
        candidate_formal_charge=candidate_charge,
        candidate_charged_atom_count=candidate_charged_atoms,
        candidate_hydrogen_count=candidate_hydrogens,
        hydrogen_delta=candidate_hydrogens - observed_hydrogens,
        candidate_changed=observed_smiles != candidate_smiles,
        candidate_method=candidate_method,
        candidate_status=candidate_status,
        local_candidate_status=local_candidates.status,
        local_candidate_smiles=local_candidates.candidate_smiles,
        charge_parent_matches_local_candidate=(
            charge_parent_matches_local_candidate
        ),
        fragment_count=fragment_count,
        heavy_atom_count=observed.GetNumHeavyAtoms(),
        radical_electrons=radical_electrons,
        elements=elements,
        warnings=tuple(warnings),
        error=None,
    )


def standardize_exported_component(component) -> ComponentStandardization:
    """Generate a reviewable identity candidate without approving a reactant."""
    failures = []
    for source, value, parser in _component_representations(component):
        if not value:
            continue
        try:
            with rdBase.BlockLogs():
                mol = parser(value)
                return _standardize_parsed_component(mol, source, failures)
        except Exception as exc:  # noqa: BLE001 - try the next representation
            failures.append(f"{source}:{type(exc).__name__}")

    message = "no valid molecular representation"
    if failures:
        message += f" ({', '.join(failures)})"
    return ComponentStandardization(
        parse_source=None,
        observed_smiles=None,
        observed_inchi_key=None,
        observed_formal_charge=None,
        observed_charged_atom_count=None,
        observed_hydrogen_count=None,
        candidate_smiles=None,
        candidate_inchi_key=None,
        candidate_formal_charge=None,
        candidate_charged_atom_count=None,
        candidate_hydrogen_count=None,
        hydrogen_delta=None,
        candidate_changed=None,
        candidate_method=None,
        candidate_status=None,
        local_candidate_status=None,
        local_candidate_smiles=(),
        charge_parent_matches_local_candidate=None,
        fragment_count=None,
        heavy_atom_count=None,
        radical_electrons=None,
        elements=(),
        warnings=(),
        error=f"ValueError:{message}",
    )

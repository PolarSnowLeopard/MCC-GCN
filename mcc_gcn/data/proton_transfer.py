"""Enumerate atom-local reverse proton-transfer candidates."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass(frozen=True)
class LocalNeutralizationEnumeration:
    status: str
    observed_formal_charge: int
    candidate_smiles: tuple[str, ...]
    attempted_configurations: int
    rejected_configurations: int


def _enumerate_allocations(capacities, total, limit):
    allocations = []
    limit_exceeded = False

    def visit(index, remaining, current):
        nonlocal limit_exceeded
        if limit_exceeded:
            return
        if index == len(capacities):
            if remaining == 0:
                allocations.append(tuple(current))
                if len(allocations) > limit:
                    limit_exceeded = True
            return
        for count in range(min(capacities[index], remaining) + 1):
            visit(index + 1, remaining - count, [*current, count])

    visit(0, total, [])
    return allocations[:limit], limit_exceeded


def enumerate_local_neutralization_candidates(
    mol,
    *,
    max_configurations=256,
) -> LocalNeutralizationEnumeration:
    """Reverse only atom-local proton transfers needed to reach net zero."""
    observed = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    observed_charge = Chem.GetFormalCharge(observed)
    if observed_charge == 0:
        return LocalNeutralizationEnumeration(
            status="not_required",
            observed_formal_charge=0,
            candidate_smiles=(),
            attempted_configurations=0,
            rejected_configurations=0,
        )

    sites = []
    if observed_charge > 0:
        for atom in observed.GetAtoms():
            capacity = min(
                max(atom.GetFormalCharge(), 0),
                atom.GetTotalNumHs(),
            )
            if capacity:
                has_opposite_neighbor = any(
                    neighbor.GetFormalCharge() < 0
                    for neighbor in atom.GetNeighbors()
                )
                sites.append(
                    (atom.GetIdx(), capacity, has_opposite_neighbor)
                )
    else:
        for atom in observed.GetAtoms():
            capacity = max(-atom.GetFormalCharge(), 0)
            if capacity:
                has_opposite_neighbor = any(
                    neighbor.GetFormalCharge() > 0
                    for neighbor in atom.GetNeighbors()
                )
                sites.append(
                    (atom.GetIdx(), capacity, has_opposite_neighbor)
                )

    external_sites = [site for site in sites if not site[2]]
    if sum(capacity for _, capacity, _ in external_sites) >= abs(
        observed_charge
    ):
        sites = external_sites

    allocations, limit_exceeded = _enumerate_allocations(
        [capacity for _, capacity, _ in sites],
        abs(observed_charge),
        max_configurations,
    )
    if limit_exceeded:
        return LocalNeutralizationEnumeration(
            status="configuration_limit_exceeded",
            observed_formal_charge=observed_charge,
            candidate_smiles=(),
            attempted_configurations=len(allocations),
            rejected_configurations=0,
        )

    candidates = set()
    rejected = 0
    for allocation in allocations:
        editable = Chem.RWMol(observed)
        for (atom_index, _, _), proton_count in zip(sites, allocation):
            if not proton_count:
                continue
            atom = editable.GetAtomWithIdx(atom_index)
            hydrogen_count = atom.GetTotalNumHs()
            atom.SetNoImplicit(True)
            if observed_charge > 0:
                atom.SetNumExplicitHs(hydrogen_count - proton_count)
                atom.SetFormalCharge(
                    atom.GetFormalCharge() - proton_count
                )
            else:
                atom.SetNumExplicitHs(hydrogen_count + proton_count)
                atom.SetFormalCharge(
                    atom.GetFormalCharge() + proton_count
                )

        candidate = editable.GetMol()
        try:
            with rdBase.BlockLogs():
                candidate.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(candidate)
                candidate = rdMolStandardize.Cleanup(candidate)
                Chem.SanitizeMol(candidate)
            if Chem.GetFormalCharge(candidate) != 0:
                rejected += 1
                continue
            candidates.add(
                Chem.MolToSmiles(
                    candidate,
                    canonical=True,
                    isomericSmiles=True,
                )
            )
        except Exception:  # noqa: BLE001 - invalid candidates are audited
            rejected += 1

    if not candidates:
        status = "no_valid_candidate"
    elif len(candidates) == 1:
        status = "unique_candidate"
    else:
        status = "ambiguous_candidates"
    return LocalNeutralizationEnumeration(
        status=status,
        observed_formal_charge=observed_charge,
        candidate_smiles=tuple(sorted(candidates)),
        attempted_configurations=len(allocations),
        rejected_configurations=rejected,
    )

"""Order-invariant RDKit descriptor features for molecular pairs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


MOLECULE_DESCRIPTOR_NAMES = (
    "molecular_weight",
    "exact_molecular_weight",
    "heavy_atom_count",
    "hetero_atom_count",
    "formal_charge",
    "hydrogen_bond_donors",
    "hydrogen_bond_acceptors",
    "topological_polar_surface_area",
    "log_p",
    "fraction_csp3",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "aliphatic_ring_count",
    "labute_accessible_surface_area",
    "molar_refractivity",
)


def molecule_descriptors(smiles: str) -> np.ndarray:
    """Return a fixed 16-value 2D descriptor vector for one molecule."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    values = np.asarray(
        [
            Descriptors.MolWt(molecule),
            Descriptors.ExactMolWt(molecule),
            Lipinski.HeavyAtomCount(molecule),
            Lipinski.NumHeteroatoms(molecule),
            Chem.GetFormalCharge(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.RingCount(molecule),
            Lipinski.NumAromaticRings(molecule),
            Lipinski.NumAliphaticRings(molecule),
            rdMolDescriptors.CalcLabuteASA(molecule),
            Crippen.MolMR(molecule),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite RDKit descriptors for {smiles!r}")
    return values


def pair_descriptors(
    reactant_a: str,
    reactant_b: str,
    *,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Combine component descriptors without encoding component order."""

    if cache is None:
        cache = {}
    if reactant_a not in cache:
        cache[reactant_a] = molecule_descriptors(reactant_a)
    if reactant_b not in cache:
        cache[reactant_b] = molecule_descriptors(reactant_b)
    vector_a = cache[reactant_a]
    vector_b = cache[reactant_b]
    return np.concatenate(
        [
            0.5 * (vector_a + vector_b),
            np.abs(vector_a - vector_b),
        ]
    )


def featurize_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Build a dense pair-feature matrix from an iterable of SMILES pairs."""

    if cache is None:
        cache = {}
    rows = [
        pair_descriptors(reactant_a, reactant_b, cache=cache)
        for reactant_a, reactant_b in pairs
    ]
    if not rows:
        return np.empty((0, 2 * len(MOLECULE_DESCRIPTOR_NAMES)))
    return np.vstack(rows)

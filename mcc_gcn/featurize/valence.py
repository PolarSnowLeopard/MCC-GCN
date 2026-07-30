"""Compatibility helpers for RDKit atom valence APIs."""

from rdkit import Chem


def get_atom_valence(atom, mode):
    valence_type = getattr(Chem, "ValenceType", None)
    if valence_type is not None and hasattr(atom, "GetValence"):
        return atom.GetValence(getattr(valence_type, mode))
    if mode == "EXPLICIT":
        return atom.GetExplicitValence()
    if mode == "IMPLICIT":
        return atom.GetImplicitValence()
    raise ValueError(f"Unknown valence mode: {mode}")

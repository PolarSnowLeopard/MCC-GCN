import numpy as np


BOND_TYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


class Bond:

    def __init__(self, rdkit_bond):
        self.end_atom_idx = rdkit_bond.GetEndAtomIdx()
        self.begin_atom_idx = rdkit_bond.GetBeginAtomIdx()
        conf = rdkit_bond.GetOwningMol().GetConformer()
        self.end_atom_coor = list(conf.GetAtomPosition(self.end_atom_idx))
        self.begin_atom_coor = list(conf.GetAtomPosition(self.begin_atom_idx))
        self.bond_type = rdkit_bond.GetBondType().__str__()
        try:
            self.type_number = BOND_TYPES.index(self.bond_type) + 1
        except ValueError:
            self.type_number = 5
        self.length = np.linalg.norm(
            np.array(self.end_atom_coor) - np.array(self.begin_atom_coor)
        )
        self.is_ring = rdkit_bond.IsInRing()
        self.is_conjugated = rdkit_bond.GetIsConjugated()

import os
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import ccdc
from ccdc import io

from .atom import Atom
from .bond import Bond
from .descriptors import compute_descriptors
from .adjacent_tensor import AdjacentTensor
from .fingerprint import Fingerprint
from .vertex_matrix import VertexMatrix

HBondCriterion = ccdc.molecule.Molecule.HBondCriterion()


def _get_edges(rdkit_mol):
    edges = {}
    for b in rdkit_mol.GetBonds():
        edges[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = Bond(b)
    return edges


class Coformer:

    def __init__(self, mol_file, removeh=False, hb_criterion=HBondCriterion):
        self.removeh = removeh

        try:
            self.csd_mol = io.MoleculeReader(mol_file)[0]
        except Exception:
            raise ValueError(f"CCDC cannot read: {mol_file}")
        self.csd_mol = self.csd_mol.components[0]
        self.csd_atoms = self.csd_mol.atoms

        if os.path.isfile(mol_file):
            if '.sdf' in mol_file or '.mol' in mol_file:
                self.rdkit_mol = AllChem.MolFromMolFile(mol_file, removeHs=False)
            elif '.mol2' in mol_file:
                self.rdkit_mol = AllChem.MolFromMol2File(mol_file, removeHs=False)
            self.molname = mol_file.split('/')[-1].split('.')[0]
        else:
            self.rdkit_mol = AllChem.MolFromMolBlock(mol_file, removeHs=False)
            self.molname = mol_file.split('\n')[0].strip()

        if self.rdkit_mol is None:
            csd_mol_ = io.MoleculeReader(mol_file)[0].components[0]
            self.rdkit_mol = AllChem.MolFromMol2Block(
                csd_mol_.to_string('mol2'), removeHs=False
            )
            if self.rdkit_mol is None:
                raise ValueError(f"RDKit cannot read: {mol_file}")

        if len(self.rdkit_mol.GetAtoms()) != len(self.csd_mol.atoms):
            self.rdkit_mol = AllChem.MolFromMolBlock(
                self.csd_mol.to_string('sdf'), removeHs=False
            )

        if self.removeh:
            self.csd_mol.remove_hydrogens()
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)
            self.csd_atoms = self.csd_mol.atoms

        self.atoms = {}
        for ix, atom in enumerate(self.rdkit_mol.GetAtoms()):
            self.atoms[ix] = Atom(atom, self.csd_atoms[ix], hb_criterion=hb_criterion)
        self.atom_number = len(self.atoms)

    def descriptors(self, includeSandP=True, charge_model='eem2015bm'):
        return compute_descriptors(self, includeSandP=includeSandP, charge_model=charge_model)

    @property
    def AdjacentTensor(self):
        return AdjacentTensor(self.atoms, self.get_edges, self.atom_number)

    @property
    def Fingerprint(self):
        return Fingerprint(self.rdkit_mol)

    @property
    def VertexMatrix(self):
        return VertexMatrix(self.atoms)

    @property
    def get_edges(self):
        return _get_edges(self.rdkit_mol)

    @property
    def hbond_donors(self):
        donors = {}
        for ix, atom in self.atoms.items():
            if atom.feature.is_donor:
                donors[ix] = atom.get_adjHs
        return donors

    @property
    def hbond_acceptors(self):
        return [ix for ix, atom in self.atoms.items() if atom.feature.is_acceptor]

    @property
    def get_DHs(self):
        dhs = []
        for hs in self.hbond_donors.values():
            dhs.extend(hs)
        return dhs

    @property
    def get_CHs(self):
        chs = []
        for ix, atom in self.atoms.items():
            if atom.feature.symbol == 'C':
                chs.extend(atom.get_adjHs)
        return chs

    @property
    def aromatic_atoms(self):
        return [a.GetIdx() for a in self.rdkit_mol.GetAromaticAtoms()]

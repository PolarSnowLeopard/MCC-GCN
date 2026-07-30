import numpy as np
import ccdc

from .valence import get_atom_valence

HBondCriterion = ccdc.molecule.Molecule.HBondCriterion()


class AtomFeatures:
    """Extracts atom-level features from paired RDKit and CSD atom objects."""

    def __init__(self, atom, hb_criterion=HBondCriterion):
        rdkit_atom = atom.rdkit_atom
        csd_atom = atom.csd_atom

        self.coordinates = atom.rdkit_coor
        self.symbol = rdkit_atom.GetSymbol()
        self.hybridization = rdkit_atom.GetHybridization().__str__()
        self.chirality = csd_atom.chirality
        self.is_chiral = csd_atom.is_chiral
        self.explicitvalence = get_atom_valence(rdkit_atom, "EXPLICIT")
        self.implicitvalence = get_atom_valence(rdkit_atom, "IMPLICIT")
        self.totalnumHs = rdkit_atom.GetTotalNumHs()
        self.formalcharge = rdkit_atom.GetFormalCharge()
        self.radical_electrons = rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = rdkit_atom.GetIsAromatic()
        self.is_acceptor = hb_criterion.is_acceptor(csd_atom)
        self.is_donor = hb_criterion.is_donor(csd_atom)
        self.is_spiro = csd_atom.is_spiro
        self.is_cyclic = csd_atom.is_cyclic
        self.is_metal = csd_atom.is_metal
        self.atomic_weight = rdkit_atom.GetMass()
        self.atomic_number = rdkit_atom.GetAtomicNum()
        self.vdw_radius = csd_atom.vdw_radius
        self.sybyl_type = csd_atom.sybyl_type
        self.degree = rdkit_atom.GetDegree()


class Atom:
    """Wraps a paired RDKit atom and CSD atom with consistent indexing."""

    def __init__(self, rdkit_atom, csd_atom, hb_criterion=HBondCriterion):
        self.rdkit_atom = rdkit_atom
        self.csd_atom = csd_atom
        self.hb_criterion = hb_criterion
        self.idx = rdkit_atom.GetIdx()
        conf = rdkit_atom.GetOwningMol().GetConformer()
        self.rdkit_coor = np.array(list(conf.GetAtomPosition(self.idx)))
        self.index = csd_atom.index
        self.csd_coor = np.array(list(csd_atom.coordinates))
        if not np.array_equal(self.rdkit_coor, self.csd_coor):
            raise ValueError(
                f"Coordinate mismatch between CSD and RDKit atoms at index {self.idx}"
            )

    @property
    def feature(self):
        return AtomFeatures(self, hb_criterion=self.hb_criterion)

    @property
    def get_bonds(self):
        from .bond import Bond
        bonds = {}
        for bond in self.rdkit_atom.GetBonds():
            bond_key = (self.idx, bond.GetOtherAtomIdx(self.idx))
            bonds[bond_key] = Bond(bond)
        return bonds

    @property
    def get_adjHs(self):
        hs = []
        for bond in self.rdkit_atom.GetBonds():
            if bond.GetOtherAtom(self.rdkit_atom).GetSymbol() == 'H':
                hs.append(bond.GetOtherAtomIdx(self.idx))
        return hs

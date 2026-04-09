"""Pure RDKit implementation of Coformer (no CCDC dependency).

Provides RDKitCoformer as a drop-in replacement for Coformer when
working from SMILES, SDF files, or mol blocks. Compatible with
Cocrystal, VertexMatrix, and AdjacentTensor.
"""
import os
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdmolops

from .bond import Bond

_FDEF_PATH = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

_PT = Chem.GetPeriodicTable()

_METAL_ATOMIC_NUMS = (
    set(range(3, 5)) | set(range(11, 14)) | set(range(19, 32)) |
    set(range(37, 51)) | set(range(55, 84)) | set(range(87, 119))
)


def _get_donor_acceptor_atoms(mol):
    feats = _FEATURE_FACTORY.GetFeaturesForMol(mol)
    donors, acceptors = set(), set()
    for feat in feats:
        if feat.GetFamily() == 'Donor':
            donors.update(feat.GetAtomIds())
        elif feat.GetFamily() == 'Acceptor':
            acceptors.update(feat.GetAtomIds())
    return donors, acceptors


def _get_spiro_atoms(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    spiro = set()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]
            if len(shared) == 1:
                spiro.update(shared)
    return spiro


class _AtomFeatures:

    def __init__(self, rdkit_atom, coords, is_donor, is_acceptor, is_spiro):
        self.coordinates = coords
        self.symbol = rdkit_atom.GetSymbol()
        hyb = rdkit_atom.GetHybridization().__str__()
        self.hybridization = 'S' if hyb == 'UNSPECIFIED' else hyb

        cip = rdkit_atom.GetPropsAsDict().get('_CIPCode', '')
        self.chirality = cip
        self.is_chiral = cip != ''

        self.explicitvalence = rdkit_atom.GetExplicitValence()
        self.implicitvalence = rdkit_atom.GetImplicitValence()
        self.totalnumHs = rdkit_atom.GetTotalNumHs()
        self.formalcharge = rdkit_atom.GetFormalCharge()
        self.radical_electrons = rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = rdkit_atom.GetIsAromatic()
        self.is_acceptor = is_acceptor
        self.is_donor = is_donor
        self.is_spiro = is_spiro
        self.is_cyclic = rdkit_atom.IsInRing()
        self.is_metal = rdkit_atom.GetAtomicNum() in _METAL_ATOMIC_NUMS
        self.atomic_weight = rdkit_atom.GetMass()
        self.atomic_number = rdkit_atom.GetAtomicNum()
        self.vdw_radius = _PT.GetRvdw(rdkit_atom.GetAtomicNum())
        self.degree = rdkit_atom.GetDegree()


class _RDKitAtom:

    def __init__(self, rdkit_atom, is_donor, is_acceptor, is_spiro):
        self.rdkit_atom = rdkit_atom
        self._is_donor = is_donor
        self._is_acceptor = is_acceptor
        self._is_spiro = is_spiro
        self.idx = rdkit_atom.GetIdx()
        conf = rdkit_atom.GetOwningMol().GetConformer()
        self.rdkit_coor = np.array(list(conf.GetAtomPosition(self.idx)))

    @property
    def feature(self):
        return _AtomFeatures(
            self.rdkit_atom, self.rdkit_coor,
            self._is_donor, self._is_acceptor, self._is_spiro,
        )

    @property
    def get_adjHs(self):
        hs = []
        for bond in self.rdkit_atom.GetBonds():
            if bond.GetOtherAtom(self.rdkit_atom).GetSymbol() == 'H':
                hs.append(bond.GetOtherAtomIdx(self.idx))
        return hs


def _get_edges(rdkit_mol):
    edges = {}
    for b in rdkit_mol.GetBonds():
        edges[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = Bond(b)
    return edges


def _mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise ValueError(f"Failed to generate 3D conformer: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


class RDKitCoformer:
    """Pure RDKit Coformer. Accepts SMILES, SDF file path, or mol block."""

    def __init__(self, input_data, input_type='smiles', name=None):
        if input_type == 'smiles':
            self.rdkit_mol = _mol_from_smiles(input_data)
            self.molname = name or input_data
        elif input_type == 'sdf':
            self.rdkit_mol = AllChem.MolFromMolFile(input_data, removeHs=False)
            if self.rdkit_mol is None:
                raise ValueError(f"Cannot read SDF: {input_data}")
            self.molname = name or os.path.basename(input_data).split('.')[0]
        elif input_type == 'mol_block':
            self.rdkit_mol = AllChem.MolFromMolBlock(input_data, removeHs=False)
            if self.rdkit_mol is None:
                raise ValueError("Cannot parse mol block")
            self.molname = name or input_data.split('\n')[0].strip()
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        rdmolops.AssignStereochemistry(self.rdkit_mol, cleanIt=True, force=True)

        donors, acceptors = _get_donor_acceptor_atoms(self.rdkit_mol)
        spiro_atoms = _get_spiro_atoms(self.rdkit_mol)

        self.atoms = {}
        for ix, atom in enumerate(self.rdkit_mol.GetAtoms()):
            self.atoms[ix] = _RDKitAtom(
                atom,
                is_donor=(ix in donors),
                is_acceptor=(ix in acceptors),
                is_spiro=(ix in spiro_atoms),
            )
        self.atom_number = len(self.atoms)

    @property
    def get_edges(self):
        return _get_edges(self.rdkit_mol)

    @property
    def aromatic_atoms(self):
        return [a.GetIdx() for a in self.rdkit_mol.GetAromaticAtoms()]

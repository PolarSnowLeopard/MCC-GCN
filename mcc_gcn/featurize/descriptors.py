import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFreeSASA, AllChem
from openbabel import pybel
import openbabel as ob


VDW_RADII = {
    'H': 1.2, 'He': 1.4, 'Li': 1.82, 'Be': 1.7, 'B': 2.08, 'C': 1.95,
    'N': 1.85, 'O': 1.7, 'F': 1.73, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73,
    'Al': 2.05, 'Si': 2.1, 'P': 2.08, 'S': 2.0, 'Cl': 1.97, 'Ar': 1.88,
    'Br': 2.1, 'I': 2.15,
}


def _coordinate_adjusting(mol):
    """Align molecule along principal axes of inertia.
    Reference: http://sobereva.com/426
    """
    mat_coor = [[0, 1], [0, 2], [1, 2]]
    diag_coor = [[1, 2], [0, 2], [0, 1]]
    atoms = mol.GetAtoms()
    atom_coors = np.array([
        mol.GetConformer().GetAtomPosition(a.GetIdx()) for a in atoms
    ])
    wts = np.expand_dims(np.array([a.GetMass() for a in atoms]), axis=1)
    diag_val = [np.sum(wts * atom_coors[:, i] ** 2) for i in diag_coor]
    mat_val = [
        np.sum(wts * np.prod(atom_coors[:, i], axis=1)) * -1 for i in mat_coor
    ]
    imt = np.zeros([3, 3])
    for i in range(3):
        imt[i, i] = diag_val[i]
        p = mat_coor[i]
        imt[p[0], p[1]] = mat_val[i]
        imt[p[1], p[0]] = mat_val[i]
    _, eig_m = np.linalg.eig(imt)
    return atom_coors.dot(eig_m)


def _axis_lengths(mol):
    coors = _coordinate_adjusting(mol)
    axis = []
    for i in range(3):
        col = coors[:, i]
        max_idx, min_idx = int(np.argmax(col)), int(np.argmin(col))
        max_r = VDW_RADII[mol.GetAtomWithIdx(max_idx).GetSymbol()]
        min_r = VDW_RADII[mol.GetAtomWithIdx(min_idx).GetSymbol()]
        axis.append((np.max(col) + max_r) - (np.min(col) - min_r))
    return sorted(axis)


def _shape_ratios(mol):
    S, M, L = _axis_lengths(mol)
    return S / L, M / L, S / M, S


def _globularity_and_frtpsa(coformer, includeSandP=1):
    """Globularity = surface area of equivalent sphere / SASA.
    FrTPSA = TPSA / SASA.
    Uses CSD molecular_volume for the sphere volume.
    """
    mol_vol = coformer.csd_mol.molecular_volume
    r_sphere = math.pow(mol_vol * 0.75 / math.pi, 1.0 / 3)
    area_sphere = 4 * math.pi * r_sphere ** 2
    radii = rdFreeSASA.classifyAtoms(coformer.rdkit_mol)
    sasa = rdFreeSASA.CalcSASA(coformer.rdkit_mol, radii)
    globularity = area_sphere / sasa
    frtpsa = Descriptors.TPSA(coformer.rdkit_mol, includeSandP=includeSandP) / sasa
    return globularity, frtpsa


def _dipole_moment(mol, charge_model='eem2015bm'):
    mol_block = Chem.MolToMolBlock(mol)
    ob_mol = pybel.readstring('mol', mol_block)
    dipole = ob.OBChargeModel_FindType(charge_model).GetDipoleMoment(ob_mol.OBMol)
    return math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2)


def compute_descriptors(coformer, includeSandP=1, charge_model='eem2015bm'):
    """Compute 16 molecular descriptors for a single coformer."""
    mol = coformer.rdkit_mol

    S_L, M_L, S_M, S = _shape_ratios(mol)
    globularity, frtpsa = _globularity_and_frtpsa(coformer, includeSandP)
    fr_no = Descriptors.NOCount(mol) / float(mol.GetNumHeavyAtoms())
    fr_aromatic = len(mol.GetAromaticAtoms()) / float(mol.GetNumHeavyAtoms())
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rbn = Descriptors.NumRotatableBonds(Chem.RemoveHs(mol))
    dipole = _dipole_moment(mol, charge_model)
    hb_ratio = 0 if hba == 0 else hbd / hba
    hb_total = hbd + hba
    hb_diff = abs(hbd - hba)
    chiral = 0

    return np.array([
        S_L, M_L, S_M, S, globularity, frtpsa, fr_no, fr_aromatic,
        hba, hbd, rbn, dipole, hb_ratio, hb_total, hb_diff, chiral,
    ])

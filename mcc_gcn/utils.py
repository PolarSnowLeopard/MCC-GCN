import pickle
import gzip
import random
import numpy as np
import torch
import os
from io import StringIO
from pathlib import Path

import requests
import cirpy
from rdkit import Chem
from rdkit.Chem import AllChem


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_dict_compressed(dict_data, file_path):
    with gzip.open(file_path, 'wb') as f:
        pickle.dump(dict_data, f)


def load_dict_compressed(file_path):
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)


def smiles_to_standard(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_sdf_string(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        sio = StringIO()
        writer = Chem.SDWriter(sio)
        writer.write(mol)
        writer.close()
        return sio.getvalue()
    except Exception as e:
        print(f"SMILES to SDF conversion error: {e}")
        return None


def _cas_to_smiles_pubchem(cas):
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
    }
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/cids/JSON"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        cid = resp.json()['IdentifierList']['CID'][0]
        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
        resp2 = requests.get(url2, headers=headers, timeout=10)
        if resp2.status_code != 200:
            return None
        return resp2.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception:
        return None


_CAS_DICT_CACHE = None


def _load_cas_dict():
    global _CAS_DICT_CACHE
    if _CAS_DICT_CACHE is not None:
        return _CAS_DICT_CACHE
    dict_path = Path(__file__).resolve().parent.parent / 'data' / 'cas_to_smiles_dict.pkl'
    if dict_path.exists():
        with open(dict_path, 'rb') as f:
            _CAS_DICT_CACHE = pickle.load(f)
    else:
        _CAS_DICT_CACHE = {}
    return _CAS_DICT_CACHE


def cas_to_smiles(cas):
    cas_dict = _load_cas_dict()
    try:
        smiles = cas_dict.get(cas)
        if smiles is None:
            smiles = cirpy.resolve(cas, 'smiles')
        if smiles is None:
            smiles = _cas_to_smiles_pubchem(cas)
        if smiles is None:
            return None
        return smiles_to_standard(smiles)
    except Exception as e:
        print(f"Error with CAS {cas}: {e}")
        return None


def resolve_npz(data_path, mol_blocks_path=None, rebuild=False):
    """Resolve the .npz feature file for a dataset.

    Priority: existing .npz > rebuild from .csv + .pkl.gz (requires CCDC) > error.
    """
    npz_path = data_path + '.npz'
    csv_path = data_path + '.csv'

    if os.path.exists(npz_path) and not rebuild:
        print(f"Loading pre-computed features: {npz_path}")
        return npz_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Neither {npz_path} nor {csv_path} found.\n"
            f"Please download the data files first (see README)."
        )
    if mol_blocks_path is None or not os.path.exists(mol_blocks_path):
        raise FileNotFoundError(
            f"Pre-computed features {npz_path} not found, and mol blocks "
            f"({mol_blocks_path}) unavailable for rebuilding.\n"
            f"Please download the .npz files (see README), or provide "
            f"HKU_data.pkl.gz to rebuild with CCDC."
        )

    print(f"Building graph features from {csv_path} (requires CCDC) ...")
    from mcc_gcn.data.dataset import GraphDataset
    ds = GraphDataset(csv_path, mol_blocks_path)
    ds.make_graph_dataset(save_name=npz_path)
    return npz_path


def remove_charges_from_sdf(input_sdf):
    output_sdf = []
    for line in input_sdf.split('\n'):
        if line.startswith("M  CHG"):
            continue
        if line[-7:-1] == '      ':
            new_line = line[:-1] + '0'
            output_sdf.append(new_line)
        else:
            output_sdf.append(line)
    return '\n'.join(output_sdf)

import numpy as np


ELEMENT_SYMBOLS = ['Cl', 'N', 'P', 'Br', 'B', 'S', 'I', 'F', 'C', 'O', 'H']
HYBRIDIZATION_TYPES = ['SP2', 'SP3', 'SP', 'S', 'SP3D', 'SP3D2']
CHIRALITY_LIST = ['', 'R', 'S']

FEATURE_SPEC = {
    'symbol': ELEMENT_SYMBOLS,
    'hybridization': HYBRIDIZATION_TYPES,
    'chirality': CHIRALITY_LIST,
    'is_chiral': [False, True],
    'explicitvalence': [1, 2, 3, 4, 5, 6],
    'implicitvalence': [0, 1, 2, 3],
    'totalnumHs': [0, 1, 2, 3],
    'formalcharge': [0, 1, -1],
    'radical_electrons': [0, 1],
    'is_aromatic': [False, True],
    'is_acceptor': [False, True],
    'is_donor': [False, True],
    'is_spiro': [False, True],
    'is_cyclic': [False, True],
    'is_metal': [False, True],
    'degree': [1, 2, 3, 4],
}

ATOM_FEATURES_TO_USE = [
    'symbol', 'hybridization', 'chirality', 'is_chiral', 'is_spiro', 'is_cyclic',
    'is_aromatic', 'is_donor', 'is_acceptor', 'degree', 'vdw_radius', 'explicitvalence',
    'implicitvalence', 'totalnumHs', 'formalcharge', 'radical_electrons', 'atomic_number',
]

# First 3 features are one-hot encoded, the rest are scalar
ONE_HOT_FEATURES = ['symbol', 'hybridization', 'chirality']
SCALAR_FEATURES = ATOM_FEATURES_TO_USE[3:]


def _one_hot(value, allowable_set):
    if value not in allowable_set:
        raise ValueError(f"Value {value} not in allowable set {allowable_set}")
    return [value == s for s in allowable_set]


class VertexMatrix:

    def __init__(self, atoms):
        self.nodes = atoms
        self.node_number = len(atoms)

    def feature_matrix(self):
        rows = []
        for idx in range(self.node_number):
            feat = self.nodes[idx].feature.__dict__
            row = []
            for key in ONE_HOT_FEATURES:
                row.extend(_one_hot(feat[key], FEATURE_SPEC[key]))
            for key in SCALAR_FEATURES:
                row.append(float(feat[key]))
            rows.append(np.array(row, dtype='float32'))
        return np.array(rows)

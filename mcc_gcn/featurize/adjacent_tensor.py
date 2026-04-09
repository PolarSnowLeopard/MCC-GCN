import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix


DISTANCE_BINS = [
    (0, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4),
    (4, 4.5), (4.5, 5), (5, 5.5), (5.5, 6), (6, float("inf")),
]


def _bin_index(value):
    for idx, (lo, hi) in enumerate(DISTANCE_BINS):
        if lo <= value < hi:
            return idx
    return len(DISTANCE_BINS) - 1


def _bin_distance_tensor(atoms):
    n = len(atoms)
    tensor = np.zeros([n, 10, n])
    for i in range(n):
        c1 = np.array(atoms[i].feature.coordinates)
        for j in range(i + 1, n):
            c2 = np.array(atoms[j].feature.coordinates)
            b = _bin_index(np.linalg.norm(c1 - c2))
            tensor[i, b, j] = 1
            tensor[j, b, i] = 1
    return tensor


def _to_coo(adj_tensor):
    coo = coo_matrix(adj_tensor.sum(axis=1))
    edge_index = [coo.row, coo.col]
    edge_attr = adj_tensor[edge_index[0], :, edge_index[1]]
    return edge_index, edge_attr


class AdjacentTensor:

    def __init__(self, atoms, edges, atom_number):
        self.atom_number = atom_number
        self.atoms = atoms
        self.edges = edges

    def _base_covalent(self, n_channels=4):
        A = np.zeros([self.atom_number, n_channels, self.atom_number])
        for (i, j), edge in self.edges.items():
            t = edge.type_number
            A[i, t - 1, j] = 1
            A[j, t - 1, i] = 1
        return A

    def OnlyCovalentBond(self, with_coo=False):
        A = self._base_covalent(4)
        return _to_coo(A) if with_coo else A

    def WithBondLenth(self, with_coo=False):
        A = np.zeros([self.atom_number, 5, self.atom_number])
        for (i, j), edge in self.edges.items():
            A[i, edge.type_number - 1, j] = 1
            A[j, edge.type_number - 1, i] = 1
            A[i, 4, j] = edge.length
            A[j, 4, i] = edge.length
        return _to_coo(A) if with_coo else A

    def WithDistanceMatrix(self, with_coo=False):
        coords = np.array([self.atoms[i].feature.coordinates for i in self.atoms])
        dist = np.expand_dims(squareform(pdist(coords, metric='euclidean')), 1)
        A = self._base_covalent(4)
        combined = np.concatenate([A, dist], axis=1)
        return _to_coo(combined) if with_coo else combined

    def WithRingAndConjugated(self, with_coo=False):
        A = np.zeros([self.atom_number, 6, self.atom_number])
        for (i, j), edge in self.edges.items():
            A[i, edge.type_number - 1, j] = 1
            A[j, edge.type_number - 1, i] = 1
            A[i, 4, j] = int(edge.is_conjugated)
            A[j, 4, i] = int(edge.is_conjugated)
            A[i, 5, j] = int(edge.is_ring)
            A[j, 5, i] = int(edge.is_ring)
        return _to_coo(A) if with_coo else A

    def AllFeature(self, with_coo=False):
        coords = np.array([self.atoms[i].feature.coordinates for i in self.atoms])
        dist = np.expand_dims(squareform(pdist(coords, metric='euclidean')), 1)
        A = np.zeros([self.atom_number, 6, self.atom_number])
        for (i, j), edge in self.edges.items():
            A[i, edge.type_number - 1, j] = 1
            A[j, edge.type_number - 1, i] = 1
            A[i, 4, j] = int(edge.is_conjugated)
            A[j, 4, i] = int(edge.is_conjugated)
            A[i, 5, j] = int(edge.is_ring)
            A[j, 5, i] = int(edge.is_ring)
        combined = np.concatenate([A, dist], axis=1)
        return _to_coo(combined) if with_coo else combined

    def WithBinDistanceMatrix(self, with_coo=False):
        bin_dist = _bin_distance_tensor(self.atoms)
        A = self._base_covalent(4)
        combined = np.concatenate([A, bin_dist], axis=1)
        return _to_coo(combined) if with_coo else combined

    def AllFeatureBin(self, with_coo=False):
        bin_dist = _bin_distance_tensor(self.atoms)
        A = np.zeros([self.atom_number, 6, self.atom_number])
        for (i, j), edge in self.edges.items():
            A[i, edge.type_number - 1, j] = 1
            A[j, edge.type_number - 1, i] = 1
            A[i, 4, j] = int(edge.is_conjugated)
            A[j, 4, i] = int(edge.is_conjugated)
            A[i, 5, j] = int(edge.is_ring)
            A[j, 5, i] = int(edge.is_ring)
        combined = np.concatenate([A, bin_dist], axis=1)
        return _to_coo(combined) if with_coo else combined

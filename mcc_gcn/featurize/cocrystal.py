import numpy as np
from scipy.sparse import coo_matrix

from .adjacent_tensor import AdjacentTensor
from .vertex_matrix import VertexMatrix


def _combine_pairs(list_a, list_b):
    if not list_a or not list_b:
        return []
    return [[a, b] for a in list_a for b in list_b]


def _interaction_adj_matrix(node_count, interactions):
    mat = np.zeros([node_count, 1, node_count])
    for i, j in interactions:
        mat[i, 0, j] = 1
        mat[j, 0, i] = 1
    return mat


class Cocrystal:

    def __init__(self, coformer1, coformer2):
        self.coformer1 = coformer1
        self.coformer2 = coformer2
        self.node_num1 = len(coformer1.atoms)
        self.node_num2 = len(coformer2.atoms)
        self.NodeNumber = self.node_num1 + self.node_num2
        self.name = f"{coformer1.molname}&{coformer2.molname}"

    @property
    def AdjacentTensor(self):
        return AdjacentTensor(self.get_nodes, self.get_edges, self.NodeNumber)

    @property
    def VertexMatrix(self):
        return VertexMatrix(self.get_nodes)

    @property
    def get_nodes(self):
        nodes = {}
        for ix, atom in self.coformer1.atoms.items():
            nodes[ix] = atom
        for ix, atom in self.coformer2.atoms.items():
            nodes[ix + self.node_num1] = atom
        return nodes

    @property
    def get_edges(self):
        edges = {}
        for k, v in self.coformer1.get_edges.items():
            edges[k] = v
        for (i, j), v in self.coformer2.get_edges.items():
            edges[(i + self.node_num1, j + self.node_num1)] = v
        return edges

    @property
    def possible_hbonds(self):
        d1_hs = []
        for hs in self.coformer1.hbond_donors.values():
            d1_hs.extend(hs)
        a1 = self.coformer1.hbond_acceptors

        d2_hs = []
        for hs in self.coformer2.hbond_donors.values():
            d2_hs.extend(hs)
        d2_hs = [h + self.node_num1 for h in d2_hs]
        a2 = [a + self.node_num1 for a in self.coformer2.hbond_acceptors]

        return _combine_pairs(d1_hs, a2) + _combine_pairs(d2_hs, a1)

    @property
    def possible_interaction(self):
        chs1 = self.coformer1.get_CHs
        chs2 = [h + self.node_num1 for h in self.coformer2.get_CHs]
        a1 = self.coformer1.hbond_acceptors
        a2 = [a + self.node_num1 for a in self.coformer2.hbond_acceptors]
        return _combine_pairs(chs1, a2) + _combine_pairs(chs2, a1)

    @property
    def possible_pipi_stack(self):
        ar1 = self.coformer1.aromatic_atoms
        ar2 = [a + self.node_num1 for a in self.coformer2.aromatic_atoms]
        return _combine_pairs(ar1, ar2)

    def InteractionTensor(self, hbond=True, pipi_stack=False, contact=False):
        if not (hbond or pipi_stack or contact):
            return None
        layers = []
        if hbond:
            layers.append(_interaction_adj_matrix(self.NodeNumber, self.possible_hbonds))
        if pipi_stack:
            layers.append(_interaction_adj_matrix(self.NodeNumber, self.possible_pipi_stack))
        if contact:
            layers.append(_interaction_adj_matrix(self.NodeNumber, self.possible_interaction))
        return np.concatenate(layers, axis=1)

    def CCGraphTensor(self, t_type='OnlyCovalentBond', hbond=True, pipi_stack=False, contact=False):
        valid_types = {
            'allfeature', 'allfeaturebin', 'isringandconjugated', 'onlycovalentbond',
            'withbindistancematrix', 'withbondlenth', 'withdistancematrix',
        }
        t_lower = t_type.lower()
        if t_lower == 'withbinbistancematrix':
            t_lower = 'withbindistancematrix'
        if t_lower not in valid_types:
            raise ValueError(f"t_type must be one of {list(valid_types)}")

        ccg_ins = AdjacentTensor(self.get_nodes, self.get_edges, self.NodeNumber)
        method_map = {name.lower(): getattr(ccg_ins, name)
                      for name in dir(ccg_ins) if not name.startswith('_')}
        ccg = method_map[t_lower]()

        interaction = self.InteractionTensor(hbond=hbond, pipi_stack=pipi_stack, contact=contact)
        if interaction is None:
            return ccg
        return np.append(ccg, interaction, axis=1)

    def COO_CCGraphTensor(self, **kwargs):
        ccg = self.CCGraphTensor(**kwargs)
        coo = coo_matrix(ccg.sum(axis=1))
        edge_index = [coo.row, coo.col]
        edge_attr = ccg[edge_index[0], :, edge_index[1]]
        return edge_index, edge_attr

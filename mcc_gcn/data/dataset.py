import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from ..featurize import Coformer, Cocrystal
from ..utils import load_dict_compressed


EXCEPTION_TAGS = {
    'BOVQUY', 'CEJPAK', 'GAWTON', 'GIPTAA', 'IDIGUY', 'LADBIB01',
    'PIGXUY', 'SIFBIT', 'SOJZEW', 'TOFPOW', 'QOVZIK', 'RIJNEF',
    'SIBFAK', 'SIBFEO', 'TOKGIJ', 'TOKGOP', 'TUQTEE', 'BEDZUF',
}


def _get_node_mask(graph_sizes, max_size=None):
    if max_size is None:
        max_size = np.max(graph_sizes)
    return np.array([
        np.pad(np.ones([s, 1]), ((0, max_size - s), (0, 0)), 'constant')
        for s in graph_sizes
    ], dtype=np.float32)


class GraphDataset(Dataset):
    """Builds graph-level features from CSV reaction tables + mol block dictionaries."""

    def __init__(self, table_path, mol_blocks_path):
        with open(table_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if lines and ',' in lines[0]:
            header = lines[0].strip().split(',')
            if any(h.isalpha() for h in header):
                lines = lines[1:]
        self.table = [line.strip().split(',') for line in lines if line.strip()]
        self.mol_blocks = load_dict_compressed(mol_blocks_path)
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if isinstance(sample, dict):
            return {
                k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                for k, v in sample.items()
            }
        return sample

    def _process_one(self, items):
        tag = items[4]
        block1 = self.mol_blocks[items[0]]
        block2 = self.mol_blocks[items[1]]
        try:
            c1 = Coformer(block1)
            c2 = Coformer(block2)
            cc = Cocrystal(c1, c2)
            label = int(items[3])

            result = {'tags': tag, 'labels': label}
            result['subgraph_size'] = np.array([c1.atom_number, c2.atom_number])

            if self._use_desc:
                result['global_state'] = cc.descriptors()
            if self._adj_type:
                result['A'] = cc.CCGraphTensor(
                    t_type=self._adj_type, hbond=self._hbond,
                    pipi_stack=self._pipi_stack, contact=self._contact,
                )
                result['V'] = cc.VertexMatrix.feature_matrix()
            if self._fp_type:
                result['fingerprints'] = cc.Fingerprints(
                    fp_type=self._fp_type, nBits=self._nBits, radii=self._radii,
                )
            return result
        except Exception:
            print(f"Bad input sample: {tag}, skipped.")
            return None

    def _preprocess(self, max_graph_size=None):
        graph_sizes = np.array([a.shape[0] for a in self.A]).astype(np.int32)
        largest = max_graph_size or max(graph_sizes)

        V_padded, A_padded = [], []
        for i in range(len(self.V)):
            V_padded.append(np.pad(
                self.V[i].astype(np.float32),
                ((0, largest - self.V[i].shape[0]), (0, 0)), 'constant',
            ))
            A_padded.append(np.pad(
                self.A[i].astype(np.float32),
                ((0, largest - self.A[i].shape[0]), (0, 0),
                 (0, largest - self.A[i].shape[0])), 'constant',
            ))

        self.V = np.stack(V_padded)
        self.A = np.stack(A_padded)
        self.labels = self.labels.astype(np.int32)
        if hasattr(self, 'global_state'):
            self.global_state = self.global_state.astype(np.float32)
        self.masks = _get_node_mask(graph_sizes, max_size=largest)
        self.graph_size = graph_sizes
        self.subgraph_size = self.subgraph_size.astype(np.int32)

    def make_graph_dataset(
        self, Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0,
        contact=0, max_graph_size=None, save_name=None,
    ):
        self._use_desc = Desc
        self._adj_type = A_type
        self._hbond = hbond
        self._pipi_stack = pipi_stack
        self._contact = contact
        self._fp_type = None
        self._nBits = None
        self._radii = None

        start = time.time()
        results = [
            self._process_one(items) for items in self.table
            if items[-1] not in EXCEPTION_TAGS
        ]
        results = [r for r in results if r is not None]
        if not results:
            raise ValueError("No valid data processed")

        attr_names = results[0].keys()
        attrs = {k: [] for k in attr_names}
        for sample in results:
            for key in attr_names:
                attrs[key].append(sample[key])

        for key in attrs:
            if key in ('labels', 'graph_size'):
                attrs[key] = np.array(attrs[key])
            else:
                shapes = [np.array(x).shape for x in attrs[key]]
                if len(set(str(s) for s in shapes)) == 1:
                    attrs[key] = np.array(attrs[key])
                else:
                    print(f"Warning: inconsistent shapes for {key}")

        self.__dict__.update(attrs)
        del results
        self._preprocess(max_graph_size=max_graph_size)

        if save_name:
            save_dict = {
                'V': self.V, 'A': self.A, 'labels': self.labels,
                'masks': self.masks, 'graph_size': self.graph_size,
                'tags': self.tags, 'subgraph_size': self.subgraph_size,
            }
            if hasattr(self, 'global_state'):
                save_dict['global_state'] = self.global_state
            np.savez(save_name, **save_dict)

        self.data = []
        for ix, tag in enumerate(self.tags):
            sample = {
                'V': self.V[ix], 'A': self.A[ix], 'label': self.labels[ix],
                'tag': tag, 'mask': self.masks[ix], 'graph_size': self.graph_size[ix],
                'subgraph_size': self.subgraph_size[ix],
            }
            if hasattr(self, 'global_state'):
                sample['global_state'] = self.global_state[ix]
            self.data.append(sample)

        for attr in ['V', 'A', 'labels', 'masks', 'graph_size', 'tags', 'subgraph_size']:
            self.__dict__.pop(attr, None)
        self.__dict__.pop('global_state', None)

        print(f"Elapsed Time: {time.time() - start:.2f} s")


class GraphDataLoader:
    """Loads pre-computed npz features and converts to PyG Data objects."""

    def __init__(self, npz_file=None):
        self.pyg_data = []
        if npz_file is None:
            return

        data = np.load(npz_file, allow_pickle=True)
        V_ = data['V']
        A_ = data['A']
        labels_ = data['labels']
        tags_ = data['tags']
        masks_ = data['masks']
        graph_size_ = data['graph_size']

        for ix in tqdm(range(len(tags_)), desc="Converting to PyG Data"):
            sample = {
                'V': V_[ix], 'A': A_[ix], 'label': labels_[ix],
                'tag': tags_[ix], 'mask': masks_[ix], 'graph_size': graph_size_[ix],
            }
            if 'global_state' in data:
                sample['global_state'] = data['global_state'][ix]
            if 'subgraph_size' in data:
                sample['subgraph_size'] = data['subgraph_size'][ix]

            self.pyg_data.append(self._to_pyg(sample))

    @staticmethod
    def _to_pyg(sample):
        x = torch.tensor(sample['V'], dtype=torch.float32)
        A = torch.tensor(sample['A'], dtype=torch.float32)
        edge_index = torch.nonzero(A.sum(1), as_tuple=False).t()
        edge_attr = torch.tensor(
            A[edge_index[0], :, edge_index[1]], dtype=torch.float32,
        )
        y = torch.tensor(sample['label'], dtype=torch.long)
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if sample.get('mask') is not None:
            pyg_data.mask = torch.tensor(sample['mask'], dtype=torch.float32)
        return pyg_data

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self.pyg_data, batch_size=batch_size, shuffle=shuffle)

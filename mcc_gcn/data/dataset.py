import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..utils import load_dict_compressed
from .quality import (
    PAIR_COLUMNS,
    audit_smiles,
    canonical_pair_key,
    read_pair_table,
)


def _get_node_mask(graph_sizes, max_size=None):
    if max_size is None:
        max_size = np.max(graph_sizes)
    return np.array([
        np.pad(np.ones([s, 1]), ((0, max_size - s), (0, 0)), 'constant')
        for s in graph_sizes
    ], dtype=np.float32)


class GraphDataset(Dataset):
    """Builds graph-level features from CSV reaction tables + mol block dictionaries."""

    def __init__(
        self,
        table_path,
        mol_blocks_path=None,
        *,
        feature_source="auto",
        rdkit_coordinate_mode="2d",
    ):
        table = read_pair_table(table_path)
        self.table = table[PAIR_COLUMNS].astype(str).values.tolist()
        if feature_source not in {"auto", "ccdc_molblock", "rdkit_smiles"}:
            raise ValueError(f"Unknown feature_source: {feature_source}")
        if feature_source == "auto":
            feature_source = (
                "ccdc_molblock" if mol_blocks_path else "rdkit_smiles"
            )
        if feature_source == "ccdc_molblock" and not mol_blocks_path:
            raise ValueError("ccdc_molblock requires mol_blocks_path")
        self.feature_source = feature_source
        self.rdkit_coordinate_mode = rdkit_coordinate_mode
        self.mol_blocks = (
            load_dict_compressed(mol_blocks_path)
            if feature_source == "ccdc_molblock"
            else None
        )
        self._coformer_cache = {}
        self.data = []
        self.rejections = []

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

    def _get_coformer(self, smiles):
        if smiles in self._coformer_cache:
            return self._coformer_cache[smiles]
        if self.feature_source == "rdkit_smiles":
            from ..featurize import RDKitCoformer

            coformer = RDKitCoformer(
                smiles,
                input_type="smiles",
                coordinate_mode=self.rdkit_coordinate_mode,
            )
        else:
            from ..featurize import Coformer

            coformer = Coformer(self.mol_blocks[smiles])
        self._coformer_cache[smiles] = coformer
        return coformer

    def _process_one(self, items):
        tag = items[4]
        try:
            from ..featurize import Cocrystal

            molecule_a = audit_smiles(items[0])
            molecule_b = audit_smiles(items[1])
            if molecule_a.error or molecule_b.error:
                raise ValueError(
                    "SMILES validation failed: "
                    f"A={molecule_a.error}, B={molecule_b.error}"
                )

            c1 = self._get_coformer(items[0])
            c2 = self._get_coformer(items[1])
            cc = Cocrystal(c1, c2)
            label = int(items[3])

            result = {'tags': tag, 'labels': label}
            result['pair_keys'] = canonical_pair_key(
                molecule_a.canonical_smiles,
                molecule_b.canonical_smiles,
            )
            result['subgraph_size'] = np.array([c1.atom_number, c2.atom_number])

            if self._adj_type:
                result['A'] = cc.CCGraphTensor(
                    t_type=self._adj_type, hbond=self._hbond,
                    pipi_stack=self._pipi_stack, contact=self._contact,
                )
                result['V'] = cc.VertexMatrix.feature_matrix()
            return result
        except Exception as exc:  # noqa: BLE001 - converted to a rejection record
            return {
                "_rejection": {
                    "reactant_A": items[0],
                    "reactant_B": items[1],
                    "label_str": items[2],
                    "label_int": items[3],
                    "identifier": tag,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            }

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
        self, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0,
        contact=0, max_graph_size=None, save_name=None, strict=True,
    ):
        self._adj_type = A_type
        self._hbond = hbond
        self._pipi_stack = pipi_stack
        self._contact = contact
        coordinate_dependent = A_type.lower() in {
            "allfeature",
            "allfeaturebin",
            "withbindistancematrix",
            "withbinbistancematrix",
            "withbondlenth",
            "withdistancematrix",
        }
        if (
            self.feature_source == "rdkit_smiles"
            and self.rdkit_coordinate_mode == "2d"
            and coordinate_dependent
        ):
            raise ValueError(
                f"{A_type} requires rdkit_coordinate_mode='3d'"
            )

        start = time.time()
        processed = [
            self._process_one(items)
            for items in tqdm(
                self.table,
                desc="Building graph features",
                unit="pair",
            )
        ]
        self.rejections = [
            item["_rejection"] for item in processed if "_rejection" in item
        ]
        if self.rejections and strict:
            examples = "; ".join(
                f"{item['identifier'] or '<no identifier>'}: "
                f"{item['error_type']} ({item['error']})"
                for item in self.rejections[:3]
            )
            raise ValueError(
                f"Feature construction rejected {len(self.rejections)} "
                f"of {len(self.table)} rows. Examples: {examples}"
            )
        results = [item for item in processed if "_rejection" not in item]
        if not results:
            raise ValueError("No valid data processed")

        attr_names = results[0].keys()
        attrs = {k: [] for k in attr_names}
        for sample in results:
            for key in attr_names:
                attrs[key].append(sample[key])

        for key, values in attrs.items():
            if key in ('labels', 'graph_size'):
                attrs[key] = np.array(values)
            else:
                shapes = [np.array(x).shape for x in values]
                if len({str(shape) for shape in shapes}) == 1:
                    attrs[key] = np.array(values)
                else:
                    print(f"Warning: inconsistent shapes for {key}")

        self.__dict__.update(attrs)
        del results
        self._preprocess(max_graph_size=max_graph_size)

        if save_name:
            save_dict = {
                'V': self.V, 'A': self.A, 'labels': self.labels,
                'masks': self.masks, 'graph_size': self.graph_size,
                'tags': self.tags, 'pair_keys': self.pair_keys,
                'subgraph_size': self.subgraph_size,
            }
            if hasattr(self, 'global_state'):
                save_dict['global_state'] = self.global_state
            np.savez(save_name, **save_dict)

        self.data = []
        for ix, tag in enumerate(self.tags):
            sample = {
                'V': self.V[ix], 'A': self.A[ix], 'label': self.labels[ix],
                'tag': tag, 'mask': self.masks[ix], 'graph_size': self.graph_size[ix],
                'pair_key': self.pair_keys[ix],
                'subgraph_size': self.subgraph_size[ix],
            }
            if hasattr(self, 'global_state'):
                sample['global_state'] = self.global_state[ix]
            self.data.append(sample)

        for attr in [
            'V', 'A', 'labels', 'masks', 'graph_size', 'tags', 'pair_keys',
            'subgraph_size',
        ]:
            self.__dict__.pop(attr, None)
        self.__dict__.pop('global_state', None)

        print(f"Elapsed Time: {time.time() - start:.2f} s")


class GraphDataLoader:
    """Loads pre-computed npz features and converts to PyG Data objects."""

    def __init__(self, npz_file=None, label_mode="stored"):
        self.pyg_data = []
        if label_mode not in {"stored", "binary"}:
            raise ValueError(f"Unknown label_mode: {label_mode}")
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
            label = labels_[ix]
            if label_mode == "binary":
                label = 0 if int(label) == 0 else 1
            sample = {
                'V': V_[ix], 'A': A_[ix], 'label': labels_[ix],
                'tag': tags_[ix], 'mask': masks_[ix], 'graph_size': graph_size_[ix],
            }
            sample['label'] = label
            if 'global_state' in data:
                sample['global_state'] = data['global_state'][ix]
            if 'subgraph_size' in data:
                sample['subgraph_size'] = data['subgraph_size'][ix]
            if 'pair_keys' in data:
                sample['pair_key'] = str(data['pair_keys'][ix])

            self.pyg_data.append(self._to_pyg(sample))

    @staticmethod
    def _to_pyg(sample):
        graph_size = sample.get('graph_size')
        if graph_size is None:
            node_count = sample['V'].shape[0]
        else:
            node_count = int(np.asarray(graph_size).item())
        if node_count <= 0 or node_count > sample['V'].shape[0]:
            raise ValueError(
                f"Invalid graph_size={node_count} for padded width {sample['V'].shape[0]}"
            )

        V = sample['V'][:node_count]
        A_np = sample['A'][:node_count, :, :node_count]

        x = torch.as_tensor(V, dtype=torch.float32)
        A = torch.as_tensor(A_np, dtype=torch.float32)
        edge_index = torch.nonzero(A.sum(1), as_tuple=False).t()
        edge_attr = A[edge_index[0], :, edge_index[1]]
        y = torch.tensor(sample['label'], dtype=torch.long)
        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        if sample.get('pair_key') is not None:
            pyg_data.pair_key = sample['pair_key']
        if sample.get('mask') is not None:
            pyg_data.mask = torch.as_tensor(
                sample['mask'][:node_count], dtype=torch.float32,
            )
        pyg_data.graph_size = torch.tensor(node_count, dtype=torch.long)
        return pyg_data

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self.pyg_data, batch_size=batch_size, shuffle=shuffle)

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch_geometric.data import Batch

from mcc_gcn.data.dataset import GraphDataLoader, GraphDataset
from mcc_gcn.models.gcn import GCNNet


class PackedFeatureTest(unittest.TestCase):
    def test_packed_artifact_matches_dense_pyg_graphs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "pairs.csv"
            table.write_text(
                "reactant_A,reactant_B,label_str,label_int,identifier\n"
                "CCO,O=C(O)O,cocrystal,2,PAIR1\n"
                "CCN,c1ccccc1,salt,1,PAIR2\n",
                encoding="utf-8",
            )
            dense_path = root / "dense.npz"
            packed_path = root / "packed.npz"

            GraphDataset(
                table,
                feature_source="rdkit_smiles",
            ).make_graph_dataset(save_name=dense_path)
            GraphDataset(
                table,
                feature_source="rdkit_smiles",
            ).make_packed_graph_dataset(save_name=packed_path)

            dense = GraphDataLoader(dense_path).pyg_data
            packed = GraphDataLoader(packed_path).pyg_data
            with np.load(packed_path, allow_pickle=False) as artifact:
                self.assertEqual(
                    str(artifact["storage_format"].item()),
                    "packed_sparse_v1",
                )
                self.assertNotIn("A", artifact.files)
                self.assertEqual(len(artifact["node_ptr"]), 3)
                self.assertEqual(len(artifact["edge_ptr"]), 3)

        self.assertEqual(len(dense), len(packed))
        for expected, actual in zip(dense, packed):
            self.assertTrue(torch.equal(expected.x, actual.x))
            self.assertTrue(
                torch.equal(expected.edge_index, actual.edge_index)
            )
            self.assertTrue(
                torch.equal(expected.edge_attr, actual.edge_attr)
            )
            self.assertEqual(expected.y.item(), actual.y.item())
            self.assertEqual(expected.pair_key, actual.pair_key)
            self.assertEqual(
                expected.graph_size.item(),
                actual.graph_size.item(),
            )

        torch.manual_seed(42)
        model = GCNNet()
        model.eval()
        dense_batch = Batch.from_data_list(dense)
        packed_batch = Batch.from_data_list(packed)
        with torch.no_grad():
            dense_logits = model(
                dense_batch.x,
                dense_batch.edge_index,
                dense_batch.batch,
            )
            packed_logits = model(
                packed_batch.x,
                packed_batch.edge_index,
                packed_batch.batch,
            )
        self.assertTrue(torch.equal(dense_logits, packed_logits))


if __name__ == "__main__":
    unittest.main()

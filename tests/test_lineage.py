import unittest

import numpy as np

from mcc_gcn.data.lineage import (
    match_feature_subset,
    sample_feature_fingerprint,
)


def artifact(values, padded_width):
    count = len(values)
    vertex = np.zeros((count, padded_width, 2), dtype=np.float32)
    adjacency = np.zeros(
        (count, padded_width, 1, padded_width),
        dtype=np.float32,
    )
    graph_size = np.zeros(count, dtype=np.int32)
    for index, value in enumerate(values):
        size = len(value)
        graph_size[index] = size
        vertex[index, :size, 0] = value
        adjacency[index, :size, 0, :size] = np.eye(size)
    return {
        "V": vertex,
        "A": adjacency,
        "labels": np.arange(count, dtype=np.int32),
        "graph_size": graph_size,
        "subgraph_size": np.column_stack(
            [graph_size, np.zeros(count, dtype=np.int32)]
        ),
    }


def packed_artifact(dense):
    vertices = []
    edge_indices = []
    edge_attributes = []
    node_ptr = [0]
    edge_ptr = [0]
    for index, size in enumerate(dense["graph_size"]):
        size = int(size)
        vertex = dense["V"][index, :size]
        adjacency = dense["A"][index, :size, :, :size]
        source, target = np.nonzero(adjacency.sum(axis=1))
        edge_index = np.vstack([source, target]).astype(np.int32)
        vertices.append(vertex)
        edge_indices.append(edge_index)
        edge_attributes.append(
            adjacency[edge_index[0], :, edge_index[1]]
        )
        node_ptr.append(node_ptr[-1] + size)
        edge_ptr.append(edge_ptr[-1] + edge_index.shape[1])
    return {
        "storage_format": np.asarray("packed_sparse_v1"),
        "V": np.concatenate(vertices),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_index": np.concatenate(edge_indices, axis=1),
        "edge_attr": np.concatenate(edge_attributes),
        "edge_ptr": np.asarray(edge_ptr, dtype=np.int64),
        "labels": dense["labels"],
        "graph_size": dense["graph_size"],
        "subgraph_size": dense["subgraph_size"],
    }


class FeatureLineageTest(unittest.TestCase):
    def test_fingerprint_ignores_padding_width(self):
        narrow = artifact([[1, 2]], padded_width=2)
        wide = artifact([[1, 2]], padded_width=5)
        self.assertEqual(
            sample_feature_fingerprint(narrow, 0),
            sample_feature_fingerprint(wide, 0),
        )

    def test_recovers_subset_indices(self):
        reference = artifact(
            [[1, 2], [3, 4, 5], [6]],
            padded_width=5,
        )
        subset = {
            key: value[[0, 2]]
            for key, value in reference.items()
        }
        self.assertEqual(
            match_feature_subset(reference, subset),
            [0, 2],
        )

    def test_fingerprint_matches_packed_sparse_storage(self):
        dense = artifact([[1, 2], [3, 4, 5]], padded_width=5)
        packed = packed_artifact(dense)
        for index in range(2):
            self.assertEqual(
                sample_feature_fingerprint(dense, index),
                sample_feature_fingerprint(packed, index),
            )


if __name__ == "__main__":
    unittest.main()

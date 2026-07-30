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


if __name__ == "__main__":
    unittest.main()

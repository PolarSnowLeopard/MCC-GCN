import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.plot_kpxkpr50_embedding import (
    _cluster_metrics,
    _legacy_padded_items,
)


class KpxKprEmbeddingTest(unittest.TestCase):
    def test_legacy_loader_preserves_padding_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            vertices = np.zeros((2, 7, 34), dtype=np.float32)
            adjacency = np.zeros((2, 7, 4, 7), dtype=np.float32)
            adjacency[:, 0, 0, 1] = 1.0
            np.savez(
                path,
                V=vertices,
                A=adjacency,
                labels=np.asarray([0, 1]),
            )

            items = _legacy_padded_items(path)

        self.assertEqual(len(items), 2)
        self.assertEqual(tuple(items[0].x.shape), (7, 34))

    def test_cluster_metrics_reward_well_separated_groups(self):
        features = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
        )
        labels = np.asarray([0, 0, 1, 1])

        metrics = _cluster_metrics(features, labels)

        self.assertGreater(metrics["silhouette"], 0.9)
        self.assertLess(metrics["davies_bouldin"], 0.1)


if __name__ == "__main__":
    unittest.main()

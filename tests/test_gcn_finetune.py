import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
    import torch  # noqa: F401

    from mcc_gcn.data.dataset import GraphDataLoader
    from mcc_gcn.models.gcn import GCNNet
except ImportError:
    GCNNet = None


@unittest.skipIf(GCNNet is None, "PyTorch/PyG is not installed")
class GCNFineTuneTest(unittest.TestCase):
    def test_frozen_batch_norm_layers_stay_in_evaluation_mode(self):
        model = GCNNet()
        model.ft_setting(train_dense_layer=1)
        model.train()

        self.assertTrue(model.training)
        for layer in [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]:
            self.assertFalse(layer.training)

    def test_unfrozen_dense_batch_norm_layers_stay_trainable(self):
        model = GCNNet()
        model.ft_setting(train_dense_layer=3)
        model.train()

        for layer in [model.bn1, model.bn2, model.bn3]:
            self.assertFalse(layer.training)
        for layer in [model.bn4, model.bn5]:
            self.assertTrue(layer.training)

    def test_full_fine_tuning_unfreezes_every_parameter(self):
        model = GCNNet()
        model.ft_setting(train_dense_layer=0)
        model.train()

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        for layer in [model.bn1, model.bn2, model.bn3, model.bn4, model.bn5]:
            self.assertTrue(layer.training)

    def test_small_model_matches_historical_binary_dimensions(self):
        model = GCNNet(num_classes=2, model_size="small")

        self.assertEqual(model.conv1.out_channels, 128)
        self.assertEqual(model.conv2.out_channels, 64)
        self.assertEqual(model.conv3.out_channels, 64)
        self.assertEqual(model.fc1.out_features, 64)
        self.assertEqual(model.fc2.out_features, 32)
        self.assertEqual(model.fc_out.out_features, 2)

    def test_binary_label_view_reuses_four_class_features(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(
                path,
                V=np.zeros((4, 2, 34), dtype=np.float32),
                A=np.zeros((4, 2, 4, 2), dtype=np.float32),
                labels=np.asarray([0, 1, 2, 3], dtype=np.int64),
                tags=np.asarray(["a", "b", "c", "d"]),
                masks=np.ones((4, 2, 1), dtype=np.float32),
                graph_size=np.asarray([2, 2, 2, 2], dtype=np.int64),
                pair_keys=np.asarray(["a", "b", "c", "d"]),
            )

            dataset = GraphDataLoader(
                npz_file=path,
                label_mode="binary",
            ).pyg_data

        self.assertEqual([item.y.item() for item in dataset], [0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

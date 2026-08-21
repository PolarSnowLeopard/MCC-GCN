import unittest
from pathlib import Path

import numpy as np

from mcc_gcn.data.minoxidil_only import load_minoxidil_only_data

ROOT = Path(__file__).resolve().parents[1]


class MinoxidilOnlyExperimentTest(unittest.TestCase):
    def _load(self, preserve_padding):
        return load_minoxidil_only_data(
            fine_tune_npz=ROOT / "data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz",
            holdout_ab_npz=ROOT / "data/HKU_data_6_experiment_1.npz",
            holdout_ba_npz=ROOT / "data/HKU_data_6_experiment_2.npz",
            fine_tune_manifest=ROOT / "data/manifests/finetune-34-lock-v1.csv",
            external_manifest=ROOT / "data/manifests/external-64-split-v1.csv",
            preserve_padding=preserve_padding,
        )

    def test_reconstructs_balanced_minoxidil_and_full_external_sets(self):
        data = self._load(preserve_padding=True)

        self.assertEqual(len(data.minoxidil_items), 40)
        self.assertEqual(len(data.external_ab_items), 64)
        self.assertEqual(len(data.external_ba_items), 64)
        minoxidil_labels = [item.y.item() for item in data.minoxidil_items[:20]]
        external_labels = [item.y.item() for item in data.external_ab_items]
        self.assertEqual(np.bincount(minoxidil_labels, minlength=4).tolist(), [5] * 4)
        self.assertEqual(
            np.bincount(external_labels, minlength=4).tolist(),
            [17, 13, 9, 25],
        )
        self.assertEqual(
            len({item.pair_key for item in data.minoxidil_items}),
            20,
        )
        self.assertEqual(
            [item.pair_key for item in data.external_ab_items],
            [f"external-{index:02d}" for index in range(64)],
        )

    def test_legacy_and_trimmed_modes_are_explicit(self):
        padded = self._load(preserve_padding=True)
        trimmed = self._load(preserve_padding=False)

        self.assertEqual({item.x.shape[0] for item in padded.minoxidil_items}, {80})
        self.assertEqual(
            {item.x.shape[0] for item in padded.external_ab_items},
            {70, 80},
        )
        self.assertTrue(
            all(
                item.x.shape[0] == item.observed_graph_size
                for item in trimmed.minoxidil_items
            )
        )
        self.assertTrue(
            any(
                item.x.shape[0] != item.observed_graph_size
                for item in padded.minoxidil_items
            )
        )


if __name__ == "__main__":
    unittest.main()

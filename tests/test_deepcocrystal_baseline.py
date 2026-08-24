import unittest

import numpy as np
import pandas as pd

from scripts.run_deepcocrystal_architecture_baseline import (
    _binary_labels,
    _class_weights,
    _clean_smiles,
    _training_arrays,
)


class DeepCocrystalBaselineTest(unittest.TestCase):
    def test_binary_task_maps_every_mcc_class_to_positive(self):
        table = pd.DataFrame({"label_int": [0, 1, 2, 3]})

        labels = _binary_labels(table)

        self.assertEqual(labels.tolist(), [0, 1, 1, 1])

    def test_training_augmentation_retains_both_orders(self):
        table = pd.DataFrame(
            {
                "reactant_A": ["CCO", "CCN"],
                "reactant_B": ["O", "N"],
            }
        )
        labels = np.asarray([0, 1], dtype=np.int64)

        left, right, augmented_labels = _training_arrays(
            table,
            labels,
            variants=2,
            seed=42,
            clean_smiles=lambda value, **_: value,
        )

        self.assertEqual(len(left), 8)
        self.assertEqual(len(right), 8)
        self.assertEqual(augmented_labels.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual((left[0], right[0]), (right[2], left[2]))

    def test_class_weighting_upweights_the_minority(self):
        weights = _class_weights(np.asarray([0, 1, 1, 1]))

        self.assertGreater(weights[0], weights[1])

    def test_extended_stereo_uses_rdkit_fallback_without_dropping_row(self):
        smiles = "F[S@OH22](F)(F)F"
        report = {
            "rdkit_stereo_fallback_count": 0,
            "rdkit_stereo_fallback_examples": [],
        }

        cleaned = _clean_smiles(
            [smiles],
            clean_smiles=lambda *_args, **_kwargs: None,
            preprocessing_report=report,
        )

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(report["rdkit_stereo_fallback_count"], 1)
        self.assertEqual(
            report["rdkit_stereo_fallback_examples"][0]["smiles"],
            smiles,
        )


if __name__ == "__main__":
    unittest.main()

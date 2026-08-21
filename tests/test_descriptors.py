import unittest

import numpy as np

from mcc_gcn.data.descriptors import (
    MOLECULE_DESCRIPTOR_NAMES,
    featurize_pairs,
    molecule_descriptors,
    pair_descriptors,
)


class DescriptorFeatureTest(unittest.TestCase):
    def test_molecule_descriptor_shape_and_values(self):
        values = molecule_descriptors("CCO")

        self.assertEqual(values.shape, (len(MOLECULE_DESCRIPTOR_NAMES),))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(values[0], 40.0)

    def test_pair_features_are_order_invariant(self):
        forward = pair_descriptors("CCO", "O=C=O")
        reverse = pair_descriptors("O=C=O", "CCO")

        np.testing.assert_allclose(forward, reverse)
        self.assertEqual(forward.shape, (2 * len(MOLECULE_DESCRIPTOR_NAMES),))

    def test_batch_featurization_reuses_expected_width(self):
        values = featurize_pairs([("CCO", "O"), ("CC", "N")])

        self.assertEqual(values.shape, (2, 2 * len(MOLECULE_DESCRIPTOR_NAMES)))

    def test_invalid_smiles_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            molecule_descriptors("not-a-smiles")


if __name__ == "__main__":
    unittest.main()

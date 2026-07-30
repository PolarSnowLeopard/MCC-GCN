import unittest

import numpy as np

from mcc_gcn.models.imbalance import calculate_class_weights


class ClassWeightTest(unittest.TestCase):
    def test_inverse_frequency_upweights_minority_without_dropping_rows(self):
        weights = calculate_class_weights(
            [0] * 10 + [1] * 2,
            2,
            mode="inverse-frequency",
        )
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_effective_number_is_less_extreme_than_inverse_frequency(self):
        labels = [0] * 1000 + [1] * 10
        inverse = calculate_class_weights(
            labels,
            2,
            mode="inverse-frequency",
        )
        effective = calculate_class_weights(
            labels,
            2,
            mode="effective-number",
            beta=0.99,
        )
        self.assertLess(
            effective[1] / effective[0],
            inverse[1] / inverse[0],
        )
        np.testing.assert_allclose(effective.mean(), 1.0)

    def test_rejects_missing_class(self):
        with self.assertRaisesRegex(ValueError, "Every configured class"):
            calculate_class_weights([0, 0], 2)

    def test_default_effective_number_materially_weights_current_imbalance(self):
        weights = calculate_class_weights([0] * 1031 + [1] * 21447, 2)
        self.assertGreater(weights[0] / weights[1], 5)
        np.testing.assert_allclose(weights.mean(), 1.0)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from mcc_gcn.models.metrics import calculate_chance_baselines


class ChanceBaselineTest(unittest.TestCase):
    def test_balanced_four_class_distribution_is_one_quarter(self):
        result = calculate_chance_baselines([0, 1, 2, 3], num_classes=4)

        self.assertAlmostEqual(
            result["prevalence_weighted_expected_accuracy"],
            0.25,
        )
        self.assertAlmostEqual(result["majority_class_accuracy"], 0.25)

    def test_uses_observed_class_prevalence(self):
        labels = np.asarray([0, 0, 0, 1])
        result = calculate_chance_baselines(labels, num_classes=2)

        self.assertAlmostEqual(
            result["prevalence_weighted_expected_accuracy"],
            0.75**2 + 0.25**2,
        )
        self.assertAlmostEqual(result["majority_class_accuracy"], 0.75)
        self.assertEqual(result["class_counts"].tolist(), [3, 1])

    def test_rejects_invalid_labels(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            calculate_chance_baselines([0, 2], num_classes=2)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            calculate_chance_baselines([], num_classes=2)


if __name__ == "__main__":
    unittest.main()

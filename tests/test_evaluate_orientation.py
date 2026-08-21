import unittest

import numpy as np

from scripts.evaluate import _orientation_analysis


class OrientationAnalysisTest(unittest.TestCase):
    def test_reports_agreement_and_probability_discrepancy(self):
        labels = np.asarray([0, 1, 1])
        probabilities_ab = np.asarray(
            [[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]]
        )
        probabilities_ba = np.asarray(
            [[0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
        )

        result = _orientation_analysis(
            labels,
            probabilities_ab,
            probabilities_ba,
            2,
            ["negative", "positive"],
        )

        np.testing.assert_array_equal(
            result["predictions_ab"],
            np.asarray([0, 1, 0]),
        )
        np.testing.assert_array_equal(
            result["predictions_ba"],
            np.asarray([0, 1, 1]),
        )
        self.assertAlmostEqual(
            result["metrics"]["agreement_fraction"],
            2 / 3,
        )
        self.assertEqual(result["metrics"]["disagreement_pairs"], 1)
        self.assertAlmostEqual(
            result["metrics"]["mean_total_variation"],
            (0.2 + 0.1 + 0.5) / 3,
        )
        self.assertAlmostEqual(
            result["metrics"]["a_b"]["overall_accuracy"],
            2 / 3,
        )
        self.assertEqual(
            result["metrics"]["b_a"]["overall_accuracy"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

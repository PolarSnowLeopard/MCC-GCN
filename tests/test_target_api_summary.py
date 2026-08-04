import unittest

import pandas as pd

from scripts.summarize_target_api_predictions import _metrics


class TargetAPISummaryTest(unittest.TestCase):
    def test_single_present_class_bootstrap_counts_off_class_predictions(self):
        table = pd.DataFrame(
            {
                "True Label": [1, 1, 1, 1],
                "Predicted Label": [1, 1, 1, 0],
            }
        )

        metrics = _metrics(
            table,
            ["negative", "positive"],
            bootstrap_replicates=200,
            seed=42,
        )

        self.assertEqual(metrics["balanced_accuracy_present_classes"], 0.75)
        self.assertEqual(
            metrics["bootstrap"]["balanced_accuracy_95_ci"],
            metrics["bootstrap"]["accuracy_95_ci"],
        )


if __name__ == "__main__":
    unittest.main()

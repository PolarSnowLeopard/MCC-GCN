import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.summarize_pretrain_imbalance import (
    aggregate_results,
    collect_results,
    write_summary,
)


class PretrainImbalanceSummaryTest(unittest.TestCase):
    def _write_run(self, root, seed, validation_bacc, external_bacc):
        run = root / "binary" / "none" / f"seed-{seed}" / "pretrain"
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(
            json.dumps(
                {
                    "task": "binary",
                    "class_weighting": "none",
                    "seed": seed,
                    "validation_aggregation": "pair_probability_mean",
                }
            ),
            encoding="utf-8",
        )
        (run / "selection_result.json").write_text(
            json.dumps(
                {
                    "best_epoch": 7,
                    "epochs_completed": 10,
                    "best_validation_balanced_accuracy": validation_bacc,
                    "best_validation_loss": 0.4,
                }
            ),
            encoding="utf-8",
        )
        (run / "external_64_predictions.metrics.json").write_text(
            json.dumps(
                {
                    "task": "binary",
                    "overall_accuracy": 0.75,
                    "balanced_accuracy": external_bacc,
                    "per_class_accuracy": {
                        "negative": 0.5,
                        "positive": 1.0,
                    },
                    "confusion_matrix": [[2, 2], [0, 4]],
                }
            ),
            encoding="utf-8",
        )

    def test_collects_counts_and_aggregates_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment_profile.json").write_text(
                json.dumps(
                    {
                        "tasks": ["binary"],
                        "class_weightings": ["none"],
                        "seeds": [42, 43],
                    }
                ),
                encoding="utf-8",
            )
            self._write_run(root, 42, 0.7, 0.6)
            self._write_run(root, 43, 0.9, 0.8)

            profile, rows, missing = collect_results(root)
            self.assertEqual(profile["tasks"], ["binary"])
            self.assertEqual(missing, [])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["external_negative_true_count"], 4)
            self.assertEqual(rows[0]["external_positive_predicted_count"], 6)

            aggregated = aggregate_results(rows)
            self.assertEqual(aggregated[0]["runs"], 2)
            self.assertAlmostEqual(
                aggregated[0]["validation_balanced_accuracy_mean"],
                0.8,
            )
            outputs = write_summary(root, profile, rows, missing)
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_reports_incomplete_expected_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment_profile.json").write_text(
                json.dumps(
                    {
                        "tasks": ["binary"],
                        "class_weightings": ["none"],
                        "seeds": [42],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                collect_results(root)


if __name__ == "__main__":
    unittest.main()

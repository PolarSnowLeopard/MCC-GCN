import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.summarize_target_api_pretrain_ablation import (
    aggregate_results,
    collect_results,
    write_summary,
)


class TargetPretrainAblationSummaryTest(unittest.TestCase):
    def _write_run(self, root, seed, validation_bacc, target_bacc):
        run = root / "binary" / "undersampled-uniform" / f"seed-{seed}" / "pretrain"
        (run / "target-summary").mkdir(parents=True)
        (run / "run_config.json").write_text(
            json.dumps(
                {
                    "task": "binary",
                    "class_weighting": "none",
                    "seed": seed,
                    "resolved_npz": str(root / "binary.npz"),
                }
            ),
            encoding="utf-8",
        )
        (run / "selection_result.json").write_text(
            json.dumps(
                {
                    "best_epoch": 7,
                    "epochs_completed": 12,
                    "best_validation_balanced_accuracy": validation_bacc,
                    "best_validation_loss": 0.4,
                }
            ),
            encoding="utf-8",
        )
        (run / "target_predictions.metrics.json").write_text(
            json.dumps({"task": "binary"}),
            encoding="utf-8",
        )
        (run / "target-summary" / "summary_metrics.json").write_text(
            json.dumps(
                {
                    "pooled": {
                        "accuracy": 0.75,
                        "balanced_accuracy_present_classes": target_bacc,
                        "confusion_matrix": [[2, 2], [0, 4]],
                        "per_class": {
                            "negative": {
                                "support": 4,
                                "recall": 0.5,
                                "precision": 1.0,
                                "f1": 2 / 3,
                            },
                            "positive": {
                                "support": 4,
                                "recall": 1.0,
                                "precision": 2 / 3,
                                "f1": 0.8,
                            },
                        },
                    },
                    "by_target_api": {
                        "paracetamol": {
                            "accuracy": 0.75,
                            "balanced_accuracy_present_classes": target_bacc,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_collects_and_aggregates_target_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "binary.npz"
            feature.touch()
            (root / "experiment_profile.json").write_text(
                json.dumps(
                    {
                        "tasks": ["binary"],
                        "enabled_strategies": ["undersampled-uniform"],
                        "seeds": [42, 43],
                        "strategies": {
                            "undersampled-uniform": {
                                "class_weighting": "none",
                                "features_by_task": {
                                    "binary": {
                                        "path": str(feature),
                                        "physical_pairs": 6,
                                        "ordered_rows": 12,
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            self._write_run(root, 42, 0.7, 0.6)
            self._write_run(root, 43, 0.9, 0.8)

            profile, rows, missing = collect_results(root)
            self.assertEqual(profile["tasks"], ["binary"])
            self.assertEqual(missing, [])
            self.assertEqual(rows[0]["training_physical_pairs"], 6)
            self.assertEqual(rows[0]["target_positive_predicted_count"], 6)
            self.assertAlmostEqual(
                rows[0]["target_negative_false_positive_rate"],
                0.5,
            )

            aggregated = aggregate_results(rows)
            self.assertEqual(aggregated[0]["runs"], 2)
            self.assertAlmostEqual(
                aggregated[0]["source_validation_balanced_accuracy_mean"],
                0.8,
            )
            outputs = write_summary(root, profile, rows, missing)
            self.assertTrue(all(path.is_file() for path in outputs))

    def test_reports_missing_expected_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiment_profile.json").write_text(
                json.dumps(
                    {
                        "tasks": ["binary"],
                        "enabled_strategies": ["undersampled-uniform"],
                        "seeds": [42],
                        "strategies": {
                            "undersampled-uniform": {
                                "class_weighting": "none",
                                "features_by_task": {
                                    "binary": {
                                        "path": str(root / "missing.npz"),
                                        "physical_pairs": 6,
                                        "ordered_rows": 12,
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete"):
                collect_results(root)


if __name__ == "__main__":
    unittest.main()

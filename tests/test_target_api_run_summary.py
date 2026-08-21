import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_target_api_runs import (
    aggregate_results,
    collect_results,
)


class TargetAPIRunSummaryTest(unittest.TestCase):
    def _write_summary(self, root, task, seed, relative, accuracy, bacc):
        path = root / task / f"seed-{seed}" / relative / "summary_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "task": task,
                    "pooled": {
                        "pairs": 10,
                        "accuracy": accuracy,
                        "balanced_accuracy_present_classes": bacc,
                        "per_class": {
                            "negative": {
                                "accuracy": 0.5,
                                "f1": 0.4,
                                "support": 2,
                            }
                        },
                    },
                    "by_target_api": {
                        "Example": {
                            "accuracy": accuracy,
                            "balanced_accuracy_present_classes": bacc,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_collects_and_aggregates_learning_curve_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, accuracy in [(42, 0.6), (43, 0.8)]:
                self._write_summary(
                    root,
                    "four-class",
                    seed,
                    Path("size-8/oof-summary"),
                    accuracy,
                    0.5,
                )
                self._write_summary(
                    root,
                    "four-class",
                    seed,
                    Path("size-16/oof-summary"),
                    accuracy + 0.1,
                    0.6,
                )

            rows = collect_results([root])
            aggregate = aggregate_results(rows)

        self.assertEqual(len(rows), 4)
        size_eight = next(
            row for row in aggregate if row["fine_tuning_pairs"] == 8
        )
        self.assertAlmostEqual(size_eight["accuracy_mean"], 0.7)
        self.assertEqual(size_eight["runs"], 2)

    def test_rejects_duplicate_seed_stage_and_size(self):
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                first = Path(first_directory)
                second = Path(second_directory)
                for root in (first, second):
                    self._write_summary(
                        root,
                        "binary",
                        42,
                        Path("pretrain/target-zero-shot-summary"),
                        0.8,
                        0.7,
                    )
                with self.assertRaisesRegex(ValueError, "Duplicate result"):
                    collect_results([first, second])


if __name__ == "__main__":
    unittest.main()

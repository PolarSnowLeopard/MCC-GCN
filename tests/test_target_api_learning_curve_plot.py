import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.plot_target_api_learning_curve import collect_curve_points


class TargetAPILearningCurvePlotTest(unittest.TestCase):
    def test_combines_zero_shot_subset_and_full_points(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            formal_rows = []
            learning_rows = []
            for task in ("binary", "four-class"):
                for stage, size in (("zero_shot", 0), ("fine_tuned_full", 136)):
                    formal_rows.append(self._row(task, stage, size))
                for size in (8, 16, 32, 48):
                    learning_rows.append(
                        self._row(task, "fine_tuned_learning_curve", size)
                    )
            formal = root / "formal.json"
            learning = root / "learning.json"
            formal.write_text(
                json.dumps({"aggregate": formal_rows}),
                encoding="utf-8",
            )
            learning.write_text(
                json.dumps({"aggregate": learning_rows}),
                encoding="utf-8",
            )

            points = collect_curve_points(learning, formal)

            self.assertEqual(len(points), 12)
            self.assertEqual(
                [
                    point["fine_tuning_pairs"]
                    for point in points
                    if point["task"] == "binary"
                ],
                [0, 8, 16, 32, 48, 136],
            )
            self.assertEqual(points[0]["runs"], 3)

    def test_rejects_incomplete_curve(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal.json"
            learning = root / "learning.json"
            formal.write_text(json.dumps({"aggregate": []}), encoding="utf-8")
            learning.write_text(json.dumps({"aggregate": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected binary"):
                collect_curve_points(learning, formal)

    @staticmethod
    def _row(task, stage, size):
        return {
            "task": task,
            "stage": stage,
            "fine_tuning_pairs": size,
            "runs": 3,
            "accuracy_mean": 0.7,
            "accuracy_std": 0.01,
            "balanced_accuracy_mean": 0.6,
            "balanced_accuracy_std": 0.02,
        }


if __name__ == "__main__":
    unittest.main()

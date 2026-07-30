import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_locked_datasets.py"
FINETUNE_LOCK = ROOT / "data" / "manifests" / "finetune-34-lock-v1.csv"
EXTERNAL_LOCK = ROOT / "data" / "manifests" / "external-64-split-v1.csv"


class LockedDatasetTest(unittest.TestCase):
    def run_materializer(self, *extra_args):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output_dir = Path(temp.name) / "locked"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--finetune-lock",
                str(FINETUNE_LOCK),
                "--external-split-lock",
                str(EXTERNAL_LOCK),
                "--output-dir",
                str(output_dir),
                *extra_args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, output_dir

    def test_provisional_mode_excludes_unresolved_chemistry(self):
        result, output_dir = self.run_materializer(
            "--mode",
            "provisional-no-ccdc",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        manifest = json.loads(
            (output_dir / "locked_dataset_manifest.json").read_text()
        )
        self.assertEqual(manifest["mode"], "provisional-no-ccdc")
        self.assertFalse(manifest["eligible_for_final_reporting"])
        self.assertEqual(
            manifest["counts"],
            {
                "excluded_unresolved_physical_pairs": 15,
                "external_physical_pairs": 64,
                "fine_tune_holdout_overlap": 0,
                "fine_tune_physical_pairs": 19,
                "holdout_physical_pairs": 50,
                "minoxidil_physical_pairs": 5,
            },
        )

        fine_tune = pd.read_csv(
            output_dir
            / "four_class_provisional_finetune_19_physical_pairs.csv"
        )
        self.assertEqual(
            fine_tune.groupby("label_int").size().to_dict(),
            {0: 8, 1: 5, 2: 3, 3: 3},
        )

    def test_final_mode_still_requires_chemistry_approvals(self):
        result, _ = self.run_materializer("--mode", "final")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--approved-minoxidil-pairs is required in final mode",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()

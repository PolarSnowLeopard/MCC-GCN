import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ccdc"
    / "export_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("export_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CcdcExportTest(unittest.TestCase):
    def test_manifest_sharding_is_deterministic(self):
        rows = [{"identifier": f"ID{index}"} for index in range(7)]
        selected = MODULE.select_shard(rows, shard_index=1, num_shards=3)
        self.assertEqual(
            [row["identifier"] for row in selected],
            ["ID1", "ID4"],
        )

    def test_manifest_requires_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text("label_str\nsalt\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identifier"):
                MODULE.read_manifest(path)


if __name__ == "__main__":
    unittest.main()

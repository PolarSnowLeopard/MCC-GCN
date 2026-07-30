import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mcc_gcn.data.quality import (
    audit_pair_table,
    build_clean_pair_table,
    build_source_overlap_table,
    read_pair_table,
)


class DataQualityTest(unittest.TestCase):
    def test_reads_header_and_preserves_column_meaning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            path.write_text(
                "reactant_B,reactant_A,label_str,label_int,identifier\n"
                "O=C(O)O,CCO,cocrystal,2,TEST01\n",
                encoding="utf-8",
            )
            table = read_pair_table(path)
        self.assertEqual(table.loc[0, "reactant_A"], "CCO")
        self.assertEqual(table.loc[0, "reactant_B"], "O=C(O)O")
        self.assertEqual(table.loc[0, "source_row"], 2)

    def test_quarantines_legacy_charge_stripping_damage(self):
        table = pd.DataFrame(
            [
                {
                    "reactant_A": "[NH3]c1ccccc1",
                    "reactant_B": "O=C(O)c1ccccc1",
                    "label_str": "salt",
                    "label_int": 1,
                    "identifier": "BAD01",
                    "source_file": "legacy.csv",
                    "source_row": 1,
                }
            ]
        )
        audited = audit_pair_table(table)
        self.assertFalse(audited.loc[0, "is_clean_candidate"])
        self.assertIn("sanitization_error", audited.loc[0, "issue_codes"])

    def test_swapped_rows_collapse_after_pair_level_curation(self):
        table = pd.DataFrame(
            [
                {
                    "reactant_A": "CCO",
                    "reactant_B": "O=C(O)O",
                    "label_str": "cocrystal",
                    "label_int": 2,
                    "identifier": "PAIR01",
                    "source_file": "pairs.csv",
                    "source_row": 1,
                },
                {
                    "reactant_A": "O=C(O)O",
                    "reactant_B": "CCO",
                    "label_str": "cocrystal",
                    "label_int": 2,
                    "identifier": "PAIR01",
                    "source_file": "pairs.csv",
                    "source_row": 2,
                },
            ]
        )
        clean = build_clean_pair_table(audit_pair_table(table))
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.loc[0, "evidence_count"], 2)

    def test_pair_with_conflicting_labels_is_not_auto_resolved(self):
        table = pd.DataFrame(
            [
                {
                    "reactant_A": "CCO",
                    "reactant_B": "O=C(O)O",
                    "label_str": "failed",
                    "label_int": 0,
                    "identifier": "",
                    "source_file": "negative.csv",
                    "source_row": 1,
                },
                {
                    "reactant_A": "O=C(O)O",
                    "reactant_B": "CCO",
                    "label_str": "cocrystal",
                    "label_int": 2,
                    "identifier": "POS01",
                    "source_file": "positive.csv",
                    "source_row": 1,
                },
            ]
        )
        audited = audit_pair_table(table)
        self.assertTrue(audited["has_label_conflict"].all())
        self.assertTrue(build_clean_pair_table(audited).empty)

    def test_reports_pair_overlap_between_sources(self):
        table = pd.DataFrame(
            [
                {
                    "reactant_A": "CCO",
                    "reactant_B": "O=C(O)O",
                    "label_str": "cocrystal",
                    "label_int": 2,
                    "identifier": "PAIR01",
                    "source_file": "train.csv",
                    "source_row": 1,
                },
                {
                    "reactant_A": "O=C(O)O",
                    "reactant_B": "CCO",
                    "label_str": "cocrystal",
                    "label_int": 2,
                    "identifier": "PAIR01",
                    "source_file": "test.csv",
                    "source_row": 1,
                },
            ]
        )
        overlap = build_source_overlap_table(audit_pair_table(table))
        self.assertEqual(overlap.loc[0, "overlapping_pairs"], 1)
        self.assertEqual(overlap.loc[0, "overlap_fraction_A"], 1.0)


if __name__ == "__main__":
    unittest.main()

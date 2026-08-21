import unittest

import pandas as pd

from scripts.summarize_revision_design import _label_counts, model_parameter_rows


class RevisionDesignSummaryTest(unittest.TestCase):
    def test_normalizes_failed_and_negative_names_by_integer_label(self):
        table = pd.DataFrame(
            {
                "label_int": [0, 0, 1],
                "label_str": ["failed", "negative", "salt"],
            }
        )

        self.assertEqual(_label_counts(table), {"negative": 2, "salt": 1})

    def test_reports_historical_and_full_fine_tuning_parameter_counts(self):
        rows = {
            (row["task"], row["fine_tuning_setting"]): row
            for row in model_parameter_rows()
        }

        self.assertEqual(rows[("binary", 0)]["total_parameters"], 23906)
        self.assertEqual(rows[("binary", 0)]["trainable_parameters"], 23906)
        self.assertEqual(rows[("binary", 3)]["trainable_parameters"], 6498)
        self.assertEqual(rows[("four-class", 0)]["total_parameters"], 134340)
        self.assertEqual(
            rows[("four-class", 3)]["trainable_parameters"],
            25412,
        )


if __name__ == "__main__":
    unittest.main()

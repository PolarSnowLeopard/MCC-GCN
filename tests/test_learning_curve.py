import unittest
from types import SimpleNamespace

import pandas as pd

from mcc_gcn.data.learning_curve import (
    balanced_nested_subsets,
    filter_items_by_pair_keys,
)


class LearningCurveSubsetTest(unittest.TestCase):
    def _table(self):
        rows = []
        for label in range(4):
            for index in range(15):
                rows.append(
                    {
                        "pair_key": f"physical-{label}-{index}",
                        "representative_pair_key": f"model-{label}-{index}",
                        "label_str": f"class-{label}",
                        "label_int": label,
                    }
                )
        return pd.DataFrame(rows)

    def test_builds_deterministic_balanced_nested_subsets(self):
        first = balanced_nested_subsets(
            self._table(),
            [8, 16, 32, 48],
            seed=42,
        )
        second = balanced_nested_subsets(
            self._table(),
            [8, 16, 32, 48],
            seed=42,
        )

        for size in [8, 16, 32, 48]:
            self.assertEqual(len(first[size]), size)
            self.assertTrue(
                first[size]["label_int"].value_counts().eq(size // 4).all()
            )
            self.assertEqual(
                list(first[size]["physical_pair_key"]),
                list(second[size]["physical_pair_key"]),
            )
        self.assertTrue(
            set(first[8]["physical_pair_key"]).issubset(
                set(first[48]["physical_pair_key"])
            )
        )

    def test_filters_all_augmented_rows_for_selected_pairs(self):
        items = [
            SimpleNamespace(pair_key=key)
            for key in ["a", "a", "b", "b", "c", "c"]
        ]
        selected = filter_items_by_pair_keys(items, {"a", "c"})
        self.assertEqual(
            [item.pair_key for item in selected],
            ["a", "a", "c", "c"],
        )

    def test_rejects_unavailable_pair(self):
        items = [SimpleNamespace(pair_key="a")]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            filter_items_by_pair_keys(items, {"missing"})


if __name__ == "__main__":
    unittest.main()

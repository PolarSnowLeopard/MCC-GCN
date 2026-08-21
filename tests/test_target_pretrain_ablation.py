import unittest

import pandas as pd

from scripts.build_target_api_pretrain_ablation import (
    augment_pair_orders,
    build_undersampled_tables,
)


class TargetPretrainAblationTest(unittest.TestCase):
    def setUp(self):
        rows = []
        counts = {0: 3, 1: 5, 2: 4, 3: 2}
        for label, count in counts.items():
            for index in range(count):
                rows.append(
                    {
                        "reactant_A": f"A-{label}-{index}",
                        "reactant_B": f"B-{label}-{index}",
                        "label_int": label,
                        "pair_key": f"pair-{label}-{index}",
                    }
                )
        self.table = pd.DataFrame(rows)

    def test_builds_task_specific_balanced_subsets(self):
        subsets = build_undersampled_tables(self.table, seed=42)

        self.assertEqual(
            subsets["four-class"]["label_int"].value_counts().sort_index().tolist(),
            [2, 2, 2, 2],
        )
        binary_counts = subsets["binary"]["binary_label_int"].value_counts()
        self.assertEqual(binary_counts.sort_index().tolist(), [3, 3])
        self.assertEqual(subsets["four-class"]["pair_key"].nunique(), 8)
        self.assertEqual(subsets["binary"]["pair_key"].nunique(), 6)

    def test_sampling_is_deterministic(self):
        first = build_undersampled_tables(self.table, seed=7)
        second = build_undersampled_tables(self.table, seed=7)

        self.assertEqual(
            first["four-class"]["pair_key"].tolist(),
            second["four-class"]["pair_key"].tolist(),
        )
        self.assertEqual(
            first["binary"]["pair_key"].tolist(),
            second["binary"]["pair_key"].tolist(),
        )

    def test_augments_only_after_physical_pair_selection(self):
        physical = build_undersampled_tables(self.table, seed=42)["four-class"]
        ordered = augment_pair_orders(physical)

        self.assertEqual(len(ordered), 2 * len(physical))
        counts = ordered.groupby("pair_key")["pair_order"].nunique()
        self.assertTrue(counts.eq(2).all())
        self.assertEqual(set(ordered["pair_order"]), {"A_B", "B_A"})


if __name__ == "__main__":
    unittest.main()

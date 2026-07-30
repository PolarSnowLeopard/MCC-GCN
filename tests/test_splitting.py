import unittest

from mcc_gcn.data.splitting import grouped_stratified_split


class GroupedSplitTest(unittest.TestCase):
    def test_keeps_both_pair_orders_in_one_split(self):
        pair_keys = []
        labels = []
        for label in range(2):
            for pair_index in range(10):
                key = f"class-{label}-pair-{pair_index}"
                pair_keys.extend([key, key])
                labels.extend([label, label])

        result = grouped_stratified_split(
            labels,
            pair_keys,
            validation_fraction=0.2,
            seed=42,
        )
        train = set(result.train_indices)
        validation = set(result.validation_indices)
        for index in range(0, len(pair_keys), 2):
            self.assertEqual(index in train, index + 1 in train)
            self.assertEqual(
                index in validation,
                index + 1 in validation,
            )
        self.assertFalse(train.intersection(validation))
        self.assertEqual(
            set(result.manifest["split"]),
            {"train", "validation"},
        )

    def test_rejects_conflicting_group_labels(self):
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            grouped_stratified_split(
                [0, 1, 0, 0, 1, 1],
                ["conflict", "conflict", "a", "b", "c", "d"],
                validation_fraction=0.5,
            )


if __name__ == "__main__":
    unittest.main()

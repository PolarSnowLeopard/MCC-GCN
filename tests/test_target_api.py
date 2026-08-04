import unittest

import pandas as pd

from mcc_gcn.data.target_api import (
    TargetAPI,
    curate_target_api_experiment,
    molecule_identity,
    stratified_target_folds,
)


def pair(smiles_a, smiles_b, label, label_int, identifier):
    return {
        "reactant_A": smiles_a,
        "reactant_B": smiles_b,
        "label_str": label,
        "label_int": label_int,
        "identifier": identifier,
    }


class TargetAPICurationTest(unittest.TestCase):
    def setUp(self):
        _, connectivity = molecule_identity("CCO")
        self.targets = (TargetAPI("Example API", "0-00-0", connectivity),)

    def test_quarantines_conflicts_and_excludes_target_from_pretraining(self):
        csd = pd.DataFrame(
            [
                pair("CCO", "CC(=O)O", "cocrystal", 2, "A1"),
                pair("CC(=O)O", "OCC", "cocrystal", 2, "A2"),
                pair("CCO", "c1ccccc1", "cocrystal", 2, "B1"),
                pair("c1ccccc1", "CCO", "solvate", 4, "B2"),
                pair("CCO", "CCN", "salt", 1, "C1"),
                pair("CC(C)=O", "CCN", "cocrystal", 2, "unrelated"),
            ]
        )
        negative = pd.DataFrame(
            [pair("CCO", "n1ccccc1", "failed", 0, "N1")]
        )
        pretrain = pd.DataFrame(
            [
                pair("CCO", "CCCl", "cocrystal", 2, "excluded"),
                pair("CC(C)=O", "CCN", "cocrystal", 2, "retained"),
            ]
        )
        pretrain["pair_key"] = ["excluded", "retained"]

        result = curate_target_api_experiment(
            csd,
            negative,
            pretrain,
            targets=self.targets,
        )

        self.assertEqual(len(result.four_class_pairs), 3)
        self.assertEqual(
            result.four_class_pairs["label_str"].value_counts().to_dict(),
            {"cocrystal": 1, "salt": 1, "negative": 1},
        )
        self.assertEqual(
            result.conflicts["connectivity_pair_key"].nunique(),
            1,
        )
        self.assertEqual(len(result.pretrain_exclusions), 1)
        self.assertEqual(
            result.four_class_pretrain_pairs["identifier"].tolist(),
            ["retained"],
        )
        self.assertEqual(
            result.binary_pairs["label_int"].value_counts().to_dict(),
            {1: 2, 0: 1},
        )
        api_canonical, _ = molecule_identity("CCO")
        self.assertTrue(
            result.four_class_pairs["reactant_A"].eq(api_canonical).all()
        )


class TargetAPIFoldTest(unittest.TestCase):
    def test_assigns_every_pair_to_one_stratified_test_fold(self):
        rows = []
        for label in range(4):
            for index in range(10):
                rows.append(
                    {
                        "pair_key": f"class-{label}-pair-{index}",
                        "label_str": f"class-{label}",
                        "label_int": label,
                        "target_apis": "Example API",
                    }
                )
        table = pd.DataFrame(rows)
        folds = stratified_target_folds(table, n_splits=5, seed=42)

        self.assertEqual(len(folds), len(table))
        self.assertEqual(folds["pair_key"].nunique(), len(table))
        self.assertEqual(set(folds["test_fold"]), set(range(5)))
        counts = folds.groupby(["test_fold", "label_int"]).size()
        self.assertTrue(counts.eq(2).all())


if __name__ == "__main__":
    unittest.main()

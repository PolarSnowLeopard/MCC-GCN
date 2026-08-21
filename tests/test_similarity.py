import unittest

import pandas as pd

from mcc_gcn.data.similarity import (
    MoleculeReference,
    nearest_pretraining_pairs,
    pairwise_tanimoto,
)


class SimilarityTest(unittest.TestCase):
    def test_pairwise_table_is_symmetric_with_unit_diagonal(self):
        matrix, long_table = pairwise_tanimoto(
            [
                MoleculeReference("ethanol", "CCO", "test"),
                MoleculeReference("ethylamine", "CCN", "test"),
            ]
        )

        self.assertEqual(matrix.loc["ethanol", "ethanol"], 1.0)
        self.assertEqual(
            matrix.loc["ethanol", "ethylamine"],
            matrix.loc["ethylamine", "ethanol"],
        )
        self.assertEqual(len(long_table), 1)

    def test_pair_matching_is_orientation_invariant(self):
        target = pd.DataFrame(
            [
                {
                    "reactant_A": "CCO",
                    "reactant_B": "c1ccccc1",
                    "pair_key": "target",
                    "target_apis": "Example",
                    "label_str": "cocrystal",
                    "label_int": 2,
                }
            ]
        )
        pretraining = pd.DataFrame(
            [
                {
                    "reactant_A": "c1ccccc1",
                    "reactant_B": "CCO",
                    "pair_key": "training",
                }
            ]
        )

        result = nearest_pretraining_pairs(target, pretraining)

        self.assertEqual(result.loc[0, "pair_nearest_tanimoto"], 1.0)
        self.assertEqual(
            result.loc[0, "nearest_pair_component_assignment"],
            "reverse",
        )
        self.assertEqual(
            result.loc[0, "api_nearest_component_tanimoto"],
            1.0,
        )
        self.assertEqual(
            result.loc[0, "coformer_nearest_component_tanimoto"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd
import torch

from mcc_gcn.models.smiles_cnn import (
    DualSmilesCNN,
    SmilesPairEvaluationDataset,
    SmilesPairTrainingDataset,
    SmilesTokenizer,
    randomized_root_smiles,
    tokenize_smiles,
)


class SmilesCNNTest(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame(
            [
                {
                    "reactant_A": "CC(=O)O",
                    "reactant_B": "c1ccccc1",
                    "label_int": 2,
                }
            ]
        )
        self.tokenizer = SmilesTokenizer.from_smiles(
            ["CC(=O)O", "c1ccccc1"],
            max_length=32,
        )

    def test_tokenizer_preserves_complete_smiles(self):
        smiles = "Nc1cc(N2CCCC2)nc(N)[n+]1[O-]"
        self.assertEqual("".join(tokenize_smiles(smiles)), smiles)

    def test_rooted_smiles_preserves_molecular_identity(self):
        from rdkit import Chem

        original = "CC(=O)Oc1ccccc1C(=O)O"
        randomized = randomized_root_smiles(original, 3)
        self.assertEqual(
            Chem.MolToSmiles(Chem.MolFromSmiles(original)),
            Chem.MolToSmiles(Chem.MolFromSmiles(randomized)),
        )

    def test_training_dataset_keeps_both_pair_orders(self):
        dataset = SmilesPairTrainingDataset(
            self.table,
            self.tokenizer,
            task="four-class",
            variants=2,
        )
        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first[2].item(), 2)
        self.assertEqual(second[2].item(), 2)

    def test_evaluation_dataset_returns_both_orders(self):
        dataset = SmilesPairEvaluationDataset(
            self.table,
            self.tokenizer,
            task="four-class",
        )
        left, right, reverse_left, reverse_right, _ = dataset[0]
        self.assertTrue(torch.equal(left, reverse_right))
        self.assertTrue(torch.equal(right, reverse_left))

    def test_freezing_encoders_keeps_classifier_trainable(self):
        model = DualSmilesCNN(
            len(self.tokenizer.vocabulary),
            4,
            filters=(8, 16),
            dense_dims=(16, 8),
        )
        model.freeze_encoders()
        model.train()
        self.assertFalse(model.left_encoder.training)
        self.assertFalse(model.right_encoder.training)
        self.assertTrue(model.classifier.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.left_encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mcc_gcn.models.gcn import (
    evaluate_pair_averaged,
    resolve_validation_aggregation,
)


class IdentityLogitModel(nn.Module):
    def forward(self, x, edge_index, batch):
        return x


def graph(logits, label, pair_key):
    return Data(
        x=torch.tensor([logits], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor(label, dtype=torch.long),
        pair_key=pair_key,
    )


class PairAveragedEvaluationTest(unittest.TestCase):
    def test_resolves_complete_pair_metadata(self):
        items = [
            graph([1.0, 0.0], 0, "pair-a"),
            graph([0.0, 1.0], 0, "pair-a"),
        ]
        self.assertEqual(
            resolve_validation_aggregation(items),
            "pair_probability_mean",
        )

    def test_rejects_mixed_pair_metadata(self):
        items = [
            graph([1.0, 0.0], 0, "pair-a"),
            graph([0.0, 1.0], 0, None),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent pair_key"):
            resolve_validation_aggregation(items)

    def test_averages_orientation_probabilities_before_prediction(self):
        items = [
            graph([4.0, 0.0], 0, "pair-a"),
            graph([0.0, 3.0], 0, "pair-a"),
            graph([2.0, 0.0], 1, "pair-b"),
            graph([0.0, 4.0], 1, "pair-b"),
        ]
        labels, predictions, loss = evaluate_pair_averaged(
            IdentityLogitModel(),
            DataLoader(items, batch_size=3, shuffle=False),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

        self.assertEqual(labels, [0, 1])
        self.assertEqual(predictions, [0, 1])

        expected_probabilities = torch.stack(
            [
                torch.stack(
                    [
                        F.softmax(items[0].x[0], dim=0),
                        F.softmax(items[1].x[0], dim=0),
                    ]
                ).mean(dim=0),
                torch.stack(
                    [
                        F.softmax(items[2].x[0], dim=0),
                        F.softmax(items[3].x[0], dim=0),
                    ]
                ).mean(dim=0),
            ]
        )
        expected_loss = F.nll_loss(
            expected_probabilities.log(),
            torch.tensor(labels),
        )
        self.assertAlmostEqual(loss, expected_loss.item(), places=6)

    def test_rejects_conflicting_labels_for_one_pair(self):
        items = [
            graph([1.0, 0.0], 0, "pair-a"),
            graph([0.0, 1.0], 1, "pair-a"),
        ]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            evaluate_pair_averaged(
                IdentityLogitModel(),
                DataLoader(items, batch_size=2),
                nn.CrossEntropyLoss(),
                torch.device("cpu"),
            )

    def test_requires_both_pair_orientations(self):
        with self.assertRaisesRegex(ValueError, "1 orientations"):
            evaluate_pair_averaged(
                IdentityLogitModel(),
                DataLoader([graph([1.0, 0.0], 0, "pair-a")]),
                nn.CrossEntropyLoss(),
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()

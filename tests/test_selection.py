import unittest

from mcc_gcn.models.selection import validation_result_improved


class ValidationSelectionTest(unittest.TestCase):
    def test_prefers_higher_metric(self):
        self.assertTrue(
            validation_result_improved(0.75, 2.0, 0.5, 1.0)
        )

    def test_uses_lower_loss_to_break_metric_tie(self):
        self.assertTrue(
            validation_result_improved(0.5, 0.9, 0.5, 1.0)
        )
        self.assertFalse(
            validation_result_improved(0.5, 1.1, 0.5, 1.0)
        )

    def test_does_not_trade_metric_for_lower_loss(self):
        self.assertFalse(
            validation_result_improved(0.49, 0.1, 0.5, 1.0)
        )


if __name__ == "__main__":
    unittest.main()

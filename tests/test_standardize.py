import unittest

from mcc_gcn.data.standardize import standardize_exported_component


class ComponentStandardizationTest(unittest.TestCase):
    def test_preserves_observed_ion_and_creates_neutral_parent(self):
        result = standardize_exported_component(
            {"smiles": "[NH3+]c1ccccc1"}
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.observed_formal_charge, 1)
        self.assertEqual(result.parent_smiles, "Nc1ccccc1")
        self.assertEqual(result.parent_formal_charge, 0)
        self.assertTrue(result.parent_changed)

    def test_flags_permanently_charged_parent(self):
        result = standardize_exported_component(
            {"smiles": "C[N+](C)(C)C"}
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.parent_formal_charge, 1)
        self.assertIn("charge_parent_remains_charged", result.warnings)

    def test_reports_invalid_component_without_guessing(self):
        result = standardize_exported_component(
            {"smiles": "[NH3]c1ccccc1"}
        )
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.parent_smiles)


if __name__ == "__main__":
    unittest.main()

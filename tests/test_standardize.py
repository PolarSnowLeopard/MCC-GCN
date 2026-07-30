import unittest

from mcc_gcn.data.standardize import standardize_exported_component


class ComponentStandardizationTest(unittest.TestCase):
    def test_preserves_observed_ion_and_creates_reviewable_candidate(self):
        result = standardize_exported_component(
            {"smiles": "[NH3+]c1ccccc1"}
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.observed_formal_charge, 1)
        self.assertEqual(result.candidate_smiles, "Nc1ccccc1")
        self.assertEqual(result.candidate_formal_charge, 0)
        self.assertEqual(result.hydrogen_delta, -1)
        self.assertTrue(result.candidate_changed)
        self.assertEqual(
            result.candidate_status,
            "neutralization_candidate_only",
        )
        self.assertEqual(result.local_candidate_status, "unique_candidate")
        self.assertTrue(result.charge_parent_matches_local_candidate)
        self.assertIn(
            "candidate_requires_pair_level_validation",
            result.warnings,
        )

    def test_neutral_observation_does_not_require_charge_parent_approval(self):
        result = standardize_exported_component({"smiles": "CCO"})
        self.assertIsNone(result.error)
        self.assertEqual(result.candidate_smiles, "CCO")
        self.assertEqual(result.candidate_method, "rdkit_cleanup")
        self.assertEqual(result.candidate_status, "observed_neutral")
        self.assertEqual(result.local_candidate_status, "not_required")

    def test_zwitterion_is_preserved_as_an_observed_net_neutral(self):
        result = standardize_exported_component(
            {"smiles": "[NH3+]CC(=O)[O-]"}
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.observed_formal_charge, 0)
        self.assertEqual(result.observed_charged_atom_count, 2)
        self.assertEqual(result.candidate_status, "observed_neutral")
        self.assertIn("observed_component_has_internal_charges", result.warnings)

    def test_flags_permanently_charged_parent(self):
        result = standardize_exported_component(
            {"smiles": "C[N+](C)(C)C"}
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.candidate_formal_charge, 1)
        self.assertEqual(result.candidate_status, "unresolved_charged")
        self.assertIn("candidate_remains_charged", result.warnings)

    def test_reports_invalid_component_without_guessing(self):
        result = standardize_exported_component(
            {"smiles": "[NH3]c1ccccc1"}
        )
        self.assertIsNotNone(result.error)
        self.assertIsNone(result.candidate_smiles)


if __name__ == "__main__":
    unittest.main()

import unittest

from mcc_gcn.data.neutralization import assess_pair_candidates


class PairNeutralizationAssessmentTest(unittest.TestCase):
    def test_accepts_neutral_pair_without_neutralization(self):
        result = assess_pair_candidates(
            {"smiles": "CCO"},
            {"smiles": "O=C(O)c1ccccc1"},
        )
        self.assertEqual(result.status, "observed_neutral_pair")
        self.assertEqual(result.observed_charge_sum, 0)
        self.assertEqual(result.hydrogen_delta_sum, 0)

    def test_marks_reverse_proton_transfer_as_candidate_only(self):
        result = assess_pair_candidates(
            {"smiles": "[NH3+]c1ccccc1"},
            {"smiles": "O=C([O-])c1ccccc1"},
        )
        self.assertEqual(result.status, "balanced_monovalent_candidate")
        self.assertEqual(result.observed_charge_sum, 0)
        self.assertEqual(result.candidate_charge_sum, 0)
        self.assertEqual(result.hydrogen_delta_sum, 0)
        self.assertTrue(result.proton_transfer_consistent)

    def test_requires_stoichiometry_for_unbalanced_unique_components(self):
        result = assess_pair_candidates(
            {"smiles": "[NH3+]CC[NH3+]"},
            {"smiles": "O=C([O-])c1ccccc1"},
        )
        self.assertEqual(result.status, "stoichiometry_required")

    def test_quarantines_ambiguous_local_neutralization(self):
        result = assess_pair_candidates(
            {"smiles": "[NH3+]C(CC(=O)[O-])C(=O)[O-]"},
            {"smiles": "[NH4+]"},
        )
        self.assertEqual(result.status, "local_neutralization_ambiguous")

    def test_preserves_net_neutral_zwitterion_pair(self):
        result = assess_pair_candidates(
            {"smiles": "[NH3+]CC(=O)[O-]"},
            {"smiles": "CCO"},
        )
        self.assertEqual(result.status, "observed_neutral_pair")


if __name__ == "__main__":
    unittest.main()

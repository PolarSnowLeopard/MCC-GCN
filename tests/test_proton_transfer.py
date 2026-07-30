import unittest

from rdkit import Chem

from mcc_gcn.data.proton_transfer import (
    enumerate_local_neutralization_candidates,
)


class LocalNeutralizationEnumerationTest(unittest.TestCase):
    def test_reverses_protonation_at_charged_nitrogen(self):
        result = enumerate_local_neutralization_candidates(
            Chem.MolFromSmiles("[NH3+]c1ccccc1")
        )
        self.assertEqual(result.status, "unique_candidate")
        self.assertEqual(result.candidate_smiles, ("Nc1ccccc1",))

    def test_reverses_deprotonation_at_charged_oxygen(self):
        result = enumerate_local_neutralization_candidates(
            Chem.MolFromSmiles("O=C([O-])c1ccccc1")
        )
        self.assertEqual(result.status, "unique_candidate")
        self.assertEqual(result.candidate_smiles, ("O=C(O)c1ccccc1",))

    def test_rejects_quaternary_ammonium_without_transferable_hydrogen(self):
        result = enumerate_local_neutralization_candidates(
            Chem.MolFromSmiles("C[N+](C)(C)C")
        )
        self.assertEqual(result.status, "no_valid_candidate")
        self.assertEqual(result.candidate_smiles, ())

    def test_reports_non_equivalent_protonation_sites_as_ambiguous(self):
        result = enumerate_local_neutralization_candidates(
            Chem.MolFromSmiles(
                "[NH3+]C(CC(=O)[O-])C(=O)[O-]"
            )
        )
        self.assertEqual(result.status, "ambiguous_candidates")
        self.assertEqual(len(result.candidate_smiles), 2)

    def test_ignores_charge_separated_nitro_oxygen_when_external_site_exists(
        self,
    ):
        result = enumerate_local_neutralization_candidates(
            Chem.MolFromSmiles(
                "O=C([O-])c1ccc([N+](=O)[O-])cc1"
            )
        )
        self.assertEqual(result.status, "unique_candidate")
        self.assertEqual(
            result.candidate_smiles,
            ("O=C(O)c1ccc([N+](=O)[O-])cc1",),
        )


if __name__ == "__main__":
    unittest.main()

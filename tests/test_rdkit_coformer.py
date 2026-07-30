import unittest

from mcc_gcn.featurize import Cocrystal, RDKitCoformer


class RDKitCoformerTest(unittest.TestCase):
    def test_builds_expected_vertex_width_without_ccdc(self):
        coformer = RDKitCoformer(
            "CCO",
            coordinate_mode="2d",
        )
        matrix = Cocrystal(
            coformer,
            RDKitCoformer("O=C(O)C", coordinate_mode="2d"),
        ).VertexMatrix.feature_matrix()
        self.assertEqual(matrix.shape[1], 34)

    def test_supports_cross_component_hbond_candidates(self):
        cocrystal = Cocrystal(
            RDKitCoformer("CCO", coordinate_mode="2d"),
            RDKitCoformer("O=C(O)C", coordinate_mode="2d"),
        )
        tensor = cocrystal.CCGraphTensor(
            t_type="OnlyCovalentBond",
            hbond=True,
        )
        self.assertEqual(tensor.shape[1], 5)
        self.assertGreater(tensor[:, 4, :].sum(), 0)


if __name__ == "__main__":
    unittest.main()

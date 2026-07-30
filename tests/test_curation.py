import unittest

import pandas as pd

from mcc_gcn.data.curation import curate_pretraining_pairs
from mcc_gcn.data.quality import canonical_pair_key, read_pair_table


def csd_row(smiles_a, smiles_b, label, label_int, identifier):
    pair_key = canonical_pair_key(smiles_a, smiles_b)
    return {
        "reactant_A": smiles_a,
        "reactant_B": smiles_b,
        "label_str": label,
        "label_int": label_int,
        "identifier": identifier,
        "source_file": "csd.csv",
        "source_row": 1,
        "pair_candidate_status": "observed_neutral_pair",
        "candidate_pair_key": pair_key,
        "candidate_has_label_conflict": False,
        "A_candidate_smiles": smiles_a,
        "B_candidate_smiles": smiles_b,
        "A_candidate_inchi_key": f"A-{identifier}",
        "B_candidate_inchi_key": f"B-{identifier}",
        "A_elements": "C",
        "B_elements": "C",
    }


class CurationTest(unittest.TestCase):
    def test_quarantines_cross_source_conflict(self):
        csd = pd.DataFrame(
            [csd_row("CC", "CCC", "cocrystal", 2, "CSD1")]
        )
        negative = pd.DataFrame(
            [
                {
                    "reactant_A": "CC",
                    "reactant_B": "CCC",
                    "label_str": "failed",
                    "label_int": 0,
                    "identifier": "",
                    "source_file": "negative.csv",
                    "source_row": 1,
                }
            ]
        )
        result = curate_pretraining_pairs(csd, negative)
        self.assertEqual(len(result.four_class_pairs), 0)
        self.assertEqual(len(result.conflicts), 2)

    def test_keeps_all_evidence_but_collapses_duplicate_pair(self):
        row = csd_row("CC", "CCCC", "salt", 1, "CSD1")
        duplicate = {**row, "identifier": "CSD2", "source_row": 2}
        negative = pd.DataFrame(
            columns=[
                "reactant_A",
                "reactant_B",
                "label_str",
                "label_int",
                "identifier",
                "source_file",
                "source_row",
            ]
        )
        result = curate_pretraining_pairs(
            pd.DataFrame([row, duplicate]),
            negative,
        )
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(len(result.four_class_pairs), 1)
        self.assertEqual(
            result.four_class_pairs.iloc[0]["evidence_count"],
            2,
        )

    def test_quarantines_elements_absent_from_atom_features(self):
        row = csd_row("C[Si](C)C", "CCO", "cocrystal", 2, "SI1")
        row["A_elements"] = "C;Si"
        negative = pd.DataFrame(
            columns=[
                "reactant_A",
                "reactant_B",
                "label_str",
                "label_int",
                "identifier",
                "source_file",
                "source_row",
            ]
        )
        result = curate_pretraining_pairs(pd.DataFrame([row]), negative)
        self.assertTrue(result.four_class_pairs.empty)
        self.assertEqual(
            result.evidence.iloc[0]["curation_reasons"],
            "unsupported_elements",
        )

    def test_quarantines_hybridization_absent_from_atom_features(self):
        row = csd_row(
            "Cl[I@SP1](Cl)c1ccccc1",
            "c1ccncc1",
            "cocrystal",
            2,
            "SP2D1",
        )
        row["A_elements"] = "Cl;I;C;H"
        row["B_elements"] = "C;H;N"
        negative = pd.DataFrame(
            columns=[
                "reactant_A",
                "reactant_B",
                "label_str",
                "label_int",
                "identifier",
                "source_file",
                "source_row",
            ]
        )
        result = curate_pretraining_pairs(pd.DataFrame([row]), negative)
        self.assertTrue(result.four_class_pairs.empty)
        self.assertEqual(
            result.evidence.iloc[0]["curation_reasons"],
            "unsupported_hybridization",
        )


if __name__ == "__main__":
    unittest.main()

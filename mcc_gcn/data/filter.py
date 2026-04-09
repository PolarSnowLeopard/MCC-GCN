"""Data filtering for CSD entries. Requires CCDC Python API."""
import re
from ccdc import io
from ccdc.entry import Entry
from rdkit import Chem


class DataFilter:

    def __init__(self, solvent_smiles_path='data/common_solvent_smiles.txt'):
        self.common_solvent_smiles = []
        self.allowed_elements = {'C', 'H', 'O', 'N', 'P', 'S', 'Cl', 'Br', 'I', 'F'}
        self.pattern = re.compile(r'^[A-Za-z]+')
        self.last_sub_identifier = ""

        with open(solvent_smiles_path, "rt", encoding="utf-8") as f:
            raw = [s.strip() for s in f.readlines()]

        for i, smi in enumerate(raw):
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            raw[i] = Chem.MolToSmiles(mol)
        self.common_solvent_smiles = raw

    def reset(self):
        self.last_sub_identifier = ""

    def is_common_solvent(self, smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            return False
        return Chem.MolToSmiles(mol) in self.common_solvent_smiles

    def get_label(self, entry):
        molecule = entry.molecule
        components = molecule.components
        components_smiles = set(c.smiles for c in components)

        if any(len(c.atoms) == 1 for c in components):
            return "single_atom", -1

        if len(components_smiles) == 3 and 'O' in components_smiles:
            return "hydrate", 3
        elif len(components_smiles) == 3 and 'O' not in components_smiles:
            return "solvate", 4
        elif len(components_smiles) == 2 and not self._is_neutral(entry):
            return "salt", 1
        elif len(components_smiles) == 2:
            return "cocrystal", 2
        return "other", 5

    def _is_neutral(self, entry):
        return all(c.formal_charge == 0 for c in entry.molecule.components)

    def filter_polymorphism(self, entry):
        identifier = entry.identifier
        match = self.pattern.match(identifier)
        letters = match.group() if match else ''
        if letters == self.last_sub_identifier:
            return False
        self.last_sub_identifier = letters
        return True

    def filter_3d_and_disorder(self, entry):
        mol = entry.molecule
        return mol.smiles is not None and entry.has_3d_structure and not entry.has_disorder

    def filter_components(self, entry):
        mol = entry.molecule
        components_smiles = set(c.smiles for c in mol.components)
        if mol.smiles is None:
            return False
        if len(components_smiles) == 2 and self._no_common_solvent(entry):
            return True
        if len(components_smiles) == 3 and sum(
            self.is_common_solvent(s) for s in components_smiles
        ) == 1:
            return True
        return False

    def _no_common_solvent(self, entry):
        return all(
            not self.is_common_solvent(c.smiles) for c in entry.molecule.components
        )

    def filter_elements(self, entry):
        elements = {atom.atomic_symbol for atom in entry.molecule.atoms}
        return elements.issubset(self.allowed_elements)

    def filter_molecular_weight(self, entry, max_weight=700):
        weights = [c.molecular_weight for c in entry.molecule.components]
        return all(w < max_weight for w in weights)

    def filter_entry_by_index(self, i, entry_list):
        try:
            entry = entry_list[i]
            result = all(f(entry) for f in [
                self.filter_polymorphism,
                self.filter_3d_and_disorder,
                self.filter_components,
                self.filter_elements,
                self.filter_molecular_weight,
            ])
            label_str, label_int = self.get_label(entry) if result else ("error", -1)
            return result, i, entry.identifier, label_str, label_int
        except Exception as e:
            print(e)
            return False, i, "", "error", -1

    def filter_entry(self, entry):
        try:
            filter_result = [
                (f.__name__, f(entry)) for f in [
                    self.filter_polymorphism,
                    self.filter_components,
                    self.filter_elements,
                    self.filter_molecular_weight,
                ]
            ]
            result = all(x[1] for x in filter_result)
            label_str, label_int = self.get_label(entry) if result else ("error", -1)
            return filter_result, result, entry.identifier, label_str, label_int
        except Exception as e:
            print(e)
            return [], False, "", "error", -1

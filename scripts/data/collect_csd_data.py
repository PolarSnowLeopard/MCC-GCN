"""Step 1: Filter CSD entries and build reaction pairs with mol blocks.

Requires CCDC Python API with valid license.

Usage:
    python scripts/data/collect_csd_data.py --output data/HKU_data_allow_inorganic.csv
"""
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccdc import io
from mcc_gcn.data.filter import DataFilter
from mcc_gcn.utils import save_dict_compressed, smiles_to_sdf_string, remove_charges_from_sdf


def step1_filter(output_csv):
    """Filter all CSD entries and save identifiers + labels."""
    csd_reader = io.EntryReader("CSD")
    n = len(csd_reader)
    filt = DataFilter()
    records = []

    for i in tqdm(range(n), desc="Filtering CSD"):
        result, _, identifier, label_str, label_int = filt.filter_entry_by_index(i, csd_reader)
        if result:
            records.append({'identifier': identifier, 'label_str': label_str, 'label_int': label_int})

    df = pd.DataFrame(records)
    df = df[df['label_int'] != -1]
    df.to_csv(output_csv, index=False)
    print(f"Filtered {len(df)} entries -> {output_csv}")
    print(df.label_int.value_counts())
    return df


def step2_build_pairs(df, output_csv, output_pkl):
    """Extract SMILES pairs and mol blocks from CSD entries."""
    csd_reader = io.EntryReader("CSD")
    filt = DataFilter()
    records = []
    reactant_data = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building pairs"):
        identifier = row['identifier']
        try:
            molecule = csd_reader.molecule(identifier)
            components = molecule.components
            components_smiles = set(c.smiles for c in components)

            non_solvent = [
                c for c in components if not filt.is_common_solvent(c.smiles)
            ]
            unique_smiles = list(set(c.smiles for c in non_solvent))
            if len(unique_smiles) != 2:
                continue

            smi_a, smi_b = unique_smiles
            for smi, comp_list in [(smi_a, non_solvent), (smi_b, non_solvent)]:
                if smi not in reactant_data:
                    comp = next(c for c in comp_list if c.smiles == smi)
                    sdf = comp.to_string('sdf')
                    reactant_data[smi] = sdf

            records.append({
                'reactant_A': smi_a,
                'reactant_B': smi_b,
                'label_str': row['label_str'],
                'label_int': row['label_int'],
                'identifier': identifier,
            })
        except Exception as e:
            print(f"Error processing {identifier}: {e}")
            continue

    reactions_df = pd.DataFrame(records)
    reactions_df.to_csv(output_csv, index=False)
    save_dict_compressed(reactant_data, output_pkl)
    print(f"Built {len(records)} reaction pairs -> {output_csv}")
    print(f"Saved {len(reactant_data)} unique mol blocks -> {output_pkl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-filtered', default='data/HKU_data_allow_inorganic.csv')
    p.add_argument('--output-pairs', default='data/HKU_data_reactions.csv')
    p.add_argument('--output-molblocks', default='data/CCDC_data.pkl.gz')
    p.add_argument('--skip-filter', action='store_true', help='Skip step 1, load existing CSV')
    args = p.parse_args()

    if args.skip_filter and os.path.exists(args.output_filtered):
        df = pd.read_csv(args.output_filtered)
        print(f"Loaded {len(df)} entries from {args.output_filtered}")
    else:
        df = step1_filter(args.output_filtered)

    step2_build_pairs(df, args.output_pairs, args.output_molblocks)


if __name__ == '__main__':
    main()

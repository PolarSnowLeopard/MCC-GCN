"""Step 2: Prepare experiment data from 64 wet experiment samples.

Converts CAS numbers to SMILES, builds reaction pairs (both orderings),
and saves CSVs for evaluation.

Usage:
    python scripts/data/prepare_experiment_data.py \
        --input data/Experimental_64_250713.csv
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mcc_gcn.utils import cas_to_smiles

LABEL_DICT = {'failed': 0, 'salt': 1, 'cocrystal': 2, 'hydrate': 3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='data/Experimental_64_250713.csv')
    p.add_argument('--output-dir', type=str, default='data')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    df['label_str'] = df['classification'].copy()
    df['label_int'] = df['label_str'].map(LABEL_DICT).astype(int)

    df['reactant_A'] = df['API'].apply(cas_to_smiles)
    df['reactant_B'] = df['coformer'].apply(cas_to_smiles)

    core = df[['reactant_A', 'reactant_B', 'label_str', 'label_int']].copy()
    core['identifier'] = ''

    # Forward ordering
    out1 = os.path.join(args.output_dir, 'HKU_data_6_experiment_1.csv')
    core.to_csv(out1, index=False)
    print(f"Saved forward order -> {out1}")

    # Reverse ordering
    rev = core.copy()
    rev[['reactant_A', 'reactant_B']] = rev[['reactant_B', 'reactant_A']]
    out2 = os.path.join(args.output_dir, 'HKU_data_6_experiment_2.csv')
    rev.to_csv(out2, index=False)
    print(f"Saved reverse order -> {out2}")

    # Merged
    merged = pd.concat([core, rev], ignore_index=True)
    out3 = os.path.join(args.output_dir, 'HKU_data_6_experiment.csv')
    merged.to_csv(out3, index=False)
    print(f"Saved merged -> {out3}")

    print(f"\nLabel distribution:\n{core.value_counts('label_str')}")


if __name__ == '__main__':
    main()

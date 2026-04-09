"""Merge all data sources into a unified mol blocks dictionary.

Combines:
  - CSD positive samples (CCDC_data.pkl.gz)
  - Fine-tuning data (FT_data.pkl.gz)
  - Experiment samples
  - Negative samples from literature

Usage:
    python scripts/data/merge_datasets.py --output data/HKU_data.pkl.gz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mcc_gcn.utils import (
    load_dict_compressed, save_dict_compressed,
    smiles_to_sdf_string, remove_charges_from_sdf,
)


def _load_csv_reactants(csv_path, remove_charges=False):
    """Extract mol blocks from a CSV of SMILES pairs."""
    reactants = {}
    with open(csv_path, 'r') as f:
        lines = [line.strip().split(',') for line in f if line.strip()]
        lines = [e for e in lines if len(e) == 5]

    for e in lines:
        for smi in [e[0], e[1]]:
            if smi not in reactants:
                sdf = smiles_to_sdf_string(smi)
                if sdf and remove_charges:
                    sdf = remove_charges_from_sdf(sdf)
                reactants[smi] = sdf
    return reactants


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csd-data', default='data/CCDC_data.pkl.gz')
    p.add_argument('--ft-data', default='data/FT_data.pkl.gz')
    p.add_argument('--ft-extra', default='data/FT_data_extra.pkl.gz')
    p.add_argument('--neg-csv', default='data/HKU_data_3_real_neg_data.csv')
    p.add_argument('--exp-csv', default='data/HKU_data_6_experiment.csv')
    p.add_argument('--ft-minoxidil-csv', default='data/HKU_data_6_FT_minoxidil.csv')
    p.add_argument('--output', default='data/HKU_data.pkl.gz')
    args = p.parse_args()

    merged = {}

    if os.path.exists(args.csd_data):
        merged.update(load_dict_compressed(args.csd_data))
        print(f"Loaded CSD data: {len(merged)} entries")

    if os.path.exists(args.ft_data):
        ft = load_dict_compressed(args.ft_data)
        merged.update(ft)
        print(f"Added FT data: {len(ft)} entries")

    if os.path.exists(args.ft_extra):
        extra = load_dict_compressed(args.ft_extra)
        merged.update(extra)
        print(f"Added FT extra: {len(extra)} entries")

    if os.path.exists(args.neg_csv):
        neg = _load_csv_reactants(args.neg_csv, remove_charges=True)
        merged.update(neg)
        print(f"Added negative samples: {len(neg)} entries")

    if os.path.exists(args.exp_csv):
        exp = _load_csv_reactants(args.exp_csv, remove_charges=False)
        merged.update(exp)
        print(f"Added experiment data: {len(exp)} entries")

    save_dict_compressed(merged, args.output)
    print(f"\nTotal merged: {len(merged)} unique mol blocks -> {args.output}")


if __name__ == '__main__':
    main()

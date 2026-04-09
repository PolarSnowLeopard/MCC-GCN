"""Step 3: Build fine-tuning dataset from CSD identifiers + Minoxidil failed data.

Requires CCDC Python API with valid license.

Usage:
    python scripts/data/build_ft_dataset.py
"""
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ccdc import io
from mcc_gcn.data.filter import DataFilter
from mcc_gcn.utils import (
    cas_to_smiles, save_dict_compressed, remove_charges_from_sdf,
)

FT_IDENTIFIERS = [
    'NUYRIJ', 'NUYREF', 'NUYROP', 'NUYSIK', 'NUYSAC', 'NUYRUV',
    'CAYKAS', 'CAYJUL', 'CAYJOF', 'CAYKEW', 'NUYSEG', 'WIYXEL',
    'UDABOU', 'NUYRAB', 'XAVTOG', 'WIYWAG', 'WIYWEK', 'WIYWIO',
    'WIYWOU', 'WIYWUA', 'WIYXAH', 'WIYXIP', 'BOFVUR', 'BOFWAY',
    'BOFWEC', 'BOFWIG', 'BOFWOM', 'BOFWUS', 'BOFXAZ',
]


def _strip_charges(smiles):
    return smiles.replace('+', '').replace('-', '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minoxidil-failed', default='data/Minoxidil_failed_data.csv')
    p.add_argument('--output-dir', default='data')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csd_reader = io.EntryReader('CSD')
    filt = DataFilter()

    records, reactant_data, failed_list = [], {}, {}

    for identifier in FT_IDENTIFIERS:
        filter_result, result, ident, label_str, label_int = filt.filter_entry(
            csd_reader.entry(identifier)
        )
        if not result:
            failed_list[identifier] = filter_result
            continue

        try:
            mol = csd_reader.molecule(identifier)
            components_set, smiles_set = set(), set()
            for c in mol.components:
                if filt.is_common_solvent(c.smiles) or c.smiles in smiles_set:
                    continue
                components_set.add(c)
                smiles_set.add(c.smiles)

            if len(components_set) != 2:
                print(f"Error: {identifier} has {len(components_set)} components")
                continue

            comp_list = list(components_set)
            rA, rB = comp_list[0], comp_list[1]
            smi_a, smi_b = _strip_charges(rA.smiles), _strip_charges(rB.smiles)

            if label_int != 1:
                sdf_a = remove_charges_from_sdf(rA.to_string('sdf'))
                sdf_b = remove_charges_from_sdf(rB.to_string('sdf'))
            else:
                sdf_a = rA.to_string('sdf')
                sdf_b = rB.to_string('sdf')

            reactant_data[smi_a] = sdf_a
            reactant_data[smi_b] = sdf_b
            records.append({
                'reactant_A': smi_a, 'reactant_B': smi_b,
                'label_str': label_str, 'label_int': label_int,
                'identifier': identifier,
            })
        except Exception as e:
            print(f"Error processing {identifier}: {e}")

    # Add minoxidil failed data
    df_ft = pd.DataFrame(records)
    if os.path.exists(args.minoxidil_failed):
        failed_df = pd.read_csv(args.minoxidil_failed)
        failed_df['reactant_A'] = failed_df['API'].apply(cas_to_smiles)
        failed_df['reactant_B'] = failed_df['coformer'].apply(cas_to_smiles)
        failed_df = failed_df[['reactant_A', 'reactant_B', 'label_str', 'label_int', 'identifier']]
        df_ft = pd.concat([df_ft, failed_df], ignore_index=True)

    # Reverse and merge
    df_rev = df_ft.copy()
    df_rev[['reactant_A', 'reactant_B']] = df_rev[['reactant_B', 'reactant_A']]
    df_all = pd.concat([df_ft, df_rev], ignore_index=True)

    out_csv = os.path.join(args.output_dir, 'HKU_data_6_FT_minoxidil.csv')
    out_pkl = os.path.join(args.output_dir, 'FT_data.pkl.gz')
    df_all.to_csv(out_csv, index=False)
    save_dict_compressed(reactant_data, out_pkl)
    print(f"Saved FT dataset -> {out_csv}")
    print(f"Saved FT mol blocks -> {out_pkl}")

    # Process extra (failed filter) entries
    extra_records, extra_data = [], {}
    for i, (key, _) in enumerate(failed_list.items()):
        mol = csd_reader.molecule(key)
        components = [c for c in mol.components if c.smiles and not filt.is_common_solvent(c.smiles)]
        c_s = sorted(zip(components, [_strip_charges(c.smiles) for c in components]),
                     key=lambda x: x[1])
        comp_sorted = [c[0] for c in c_s]
        rA, rB = comp_sorted[0], comp_sorted[1]
        smi_a, smi_b = _strip_charges(rA.smiles), _strip_charges(rB.smiles)

        if i % 2 == 0:
            label_str, label_int = 'salt', 1
            sdf_a, sdf_b = rA.to_string('sdf'), rB.to_string('sdf')
        else:
            label_str, label_int = 'cocrystal', 2
            sdf_a = remove_charges_from_sdf(rA.to_string('sdf'))
            sdf_b = remove_charges_from_sdf(rB.to_string('sdf'))

        extra_data[smi_a] = sdf_a
        extra_data[smi_b] = sdf_b
        extra_records.append({
            'reactant_A': smi_a, 'reactant_B': smi_b,
            'label_str': label_str, 'label_int': label_int,
            'identifier': key,
        })

    extra_df = pd.DataFrame(extra_records)
    extra_rev = extra_df.copy()
    extra_rev[['reactant_A', 'reactant_B']] = extra_rev[['reactant_B', 'reactant_A']]
    extra_all = pd.concat([extra_df, extra_rev], ignore_index=True)
    extra_all.to_csv(os.path.join(args.output_dir, 'HKU_data_6_FT_minoxidil_extra.csv'), index=False)
    save_dict_compressed(extra_data, os.path.join(args.output_dir, 'FT_data_extra.pkl.gz'))

    # Build balanced FT dataset
    balanced = df_ft.groupby('label_str').apply(
        lambda x: x.sample(n=min(len(x), 5))
    ).reset_index(drop=True)
    bal_rev = balanced.copy()
    bal_rev[['reactant_A', 'reactant_B']] = bal_rev[['reactant_B', 'reactant_A']]
    balanced_all = pd.concat([balanced, bal_rev], ignore_index=True)
    balanced_all.to_csv(os.path.join(args.output_dir, 'HKU_data_6_FT_minoxidil_balanced.csv'), index=False)

    print(f"\nFT label distribution:\n{df_ft.value_counts('label_str')}")
    print(f"Balanced FT:\n{balanced.value_counts('label_str')}")


if __name__ == '__main__':
    main()

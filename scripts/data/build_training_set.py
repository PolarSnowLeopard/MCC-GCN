"""Build balanced training sets from the merged reaction data.

Usage:
    python scripts/data/build_training_set.py \
        --input data/HKU_data_4_reactions_total.csv \
        --mol-blocks data/HKU_data.pkl.gz \
        --output data/HKU_data_5_merge_crystal_dataset.csv
"""
import argparse
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mcc_gcn.utils import load_dict_compressed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=True, help='Merged reactions CSV')
    p.add_argument('--mol-blocks', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--big', action='store_true', help='Use larger balanced size')
    args = p.parse_args()

    random.seed(args.seed)

    reactant_data = load_dict_compressed(args.mol_blocks)
    all_smiles = list(reactant_data.keys())

    with open(args.input, 'r') as f:
        raw = f.readlines()[1:]
        entries = [line.strip().split(',') for line in raw if line.strip()]

    by_class = {str(i): [] for i in range(5)}
    random_neg = []

    for _ in range(20000):
        a, b = random.sample(all_smiles, 2)
        random_neg.append([a, b, 'failed', '0', ''])
        random_neg.append([b, a, 'failed', '0', ''])

    for e in entries:
        rev = e[1::-1] + e[2:]
        cls = e[3]
        if cls in by_class:
            by_class[cls].append(e)
            by_class[cls].append(rev)

    if args.big:
        n = (len(by_class['3']) + len(by_class['4']))
    else:
        n = min(len(by_class[c]) for c in ['0', '1', '2', '3', '4'] if by_class[c])
        n = len(by_class['0']) if by_class['0'] else 1052 * 2

    neg_total = by_class['0']
    if len(neg_total) < n:
        neg_total = neg_total + random_neg[:n - len(neg_total)]

    # Merge hydrate + solvate into class 3
    hs_combined = by_class['3'] + by_class['4']
    for e in hs_combined:
        e[3] = '3'

    dataset = []
    dataset.extend(random.sample(neg_total, min(n, len(neg_total))))
    dataset.extend(random.sample(by_class['1'], min(n, len(by_class['1']))))
    dataset.extend(random.sample(by_class['2'], min(n, len(by_class['2']))))
    dataset.extend(random.sample(hs_combined, min(n, len(hs_combined))))

    random.shuffle(dataset)

    with open(args.output, 'w') as f:
        f.write(''.join(','.join(e) + '\n' if not e[-1].endswith('\n') else ','.join(e) for e in dataset))

    print(f"Built balanced dataset with {len(dataset)} samples -> {args.output}")


if __name__ == '__main__':
    main()

"""Single-sample MCC-GCN prediction from SMILES, CAS, or SDF input.

Usage:
    python scripts/predict.py --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "OC(=O)CC(=O)O"
    python scripts/predict.py --cas "58-08-2" "141-82-2"
    python scripts/predict.py --sdf mol1.sdf mol2.sdf
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcc_gcn.models.gcn import GCNNet
from mcc_gcn.featurize.rdkit_coformer import RDKitCoformer
from mcc_gcn.featurize.cocrystal import Cocrystal

CLASS_NAMES = ['fail', 'salt', 'cocrystal', 'hydrate/solvate']


def build_pyg_data(coformer1, coformer2):
    cc = Cocrystal(coformer1, coformer2)
    V = cc.VertexMatrix.feature_matrix()
    A = cc.CCGraphTensor(t_type='OnlyCovalentBond', hbond=False, pipi_stack=False, contact=False)

    x = torch.tensor(V, dtype=torch.float32)
    A_t = torch.tensor(A, dtype=torch.float32)
    edge_index = torch.nonzero(A_t.sum(1), as_tuple=False).t()
    edge_attr = A_t[edge_index[0], :, edge_index[1]]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def resolve_inputs(args):
    if args.smiles:
        c1 = RDKitCoformer(args.smiles[0], input_type='smiles')
        c2 = RDKitCoformer(args.smiles[1], input_type='smiles')
    elif args.cas:
        from mcc_gcn.utils import cas_to_smiles
        s1 = cas_to_smiles(args.cas[0])
        s2 = cas_to_smiles(args.cas[1])
        if s1 is None:
            sys.exit(f"Error: cannot resolve CAS {args.cas[0]} to SMILES")
        if s2 is None:
            sys.exit(f"Error: cannot resolve CAS {args.cas[1]} to SMILES")
        print(f"Resolved: {args.cas[0]} -> {s1}")
        print(f"Resolved: {args.cas[1]} -> {s2}")
        c1 = RDKitCoformer(s1, input_type='smiles', name=args.cas[0])
        c2 = RDKitCoformer(s2, input_type='smiles', name=args.cas[1])
    elif args.sdf:
        c1 = RDKitCoformer(args.sdf[0], input_type='sdf')
        c2 = RDKitCoformer(args.sdf[1], input_type='sdf')
    else:
        sys.exit("Error: provide --smiles, --cas, or --sdf")
    return c1, c2


def parse_args():
    p = argparse.ArgumentParser(description="MCC-GCN Prediction")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--smiles', nargs=2, metavar=('SMILES1', 'SMILES2'))
    g.add_argument('--cas', nargs=2, metavar=('CAS1', 'CAS2'))
    g.add_argument('--sdf', nargs=2, metavar=('SDF1', 'SDF2'))
    p.add_argument('--model', type=str, default='checkpoints/best_FT_model.pth')
    p.add_argument('--large', action='store_true')
    p.add_argument('--num-classes', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    c1, c2 = resolve_inputs(args)
    data = build_pyg_data(c1, c2)

    model = GCNNet(num_classes=args.num_classes, is_large=args.large).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        output = model(data.x.to(device), data.edge_index.to(device), batch)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))

    print(f"\nMolecule 1: {c1.molname} ({c1.atom_number} atoms)")
    print(f"Molecule 2: {c2.molname} ({c2.atom_number} atoms)")
    print(f"\nPrediction: {CLASS_NAMES[pred_idx]}")
    print(f"\nConfidence:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar_len = int(prob * 30)
        bar = '#' * bar_len + '.' * (30 - bar_len)
        marker = ' <--' if i == pred_idx else ''
        print(f"  {name:>17s}  [{bar}] {prob:6.2%}{marker}")


if __name__ == '__main__':
    main()

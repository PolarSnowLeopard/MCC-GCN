"""Evaluate a trained MCC-GCN model on a test set.

Usage:
    python scripts/evaluate.py --model checkpoints/best_FT_model.pth \
        --test-data-1 HKU_data_6_experiment_1 \
        --test-data-2 HKU_data_6_experiment_2
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcc_gcn.utils import seed_everything, resolve_npz
from mcc_gcn.models.gcn import GCNNet
from mcc_gcn.data.dataset import GraphDataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MCC-GCN Evaluation")
    p.add_argument('--model', type=str, required=True, help='Path to trained model')
    p.add_argument('--test-data-1', type=str, required=True, help='Test dataset (order A-B)')
    p.add_argument('--test-data-2', type=str, required=True, help='Test dataset (order B-A)')
    p.add_argument('--mol-blocks', type=str, default='data/HKU_data.pkl.gz')
    p.add_argument('--rebuild-features', action='store_true')
    p.add_argument('--large', action='store_true')
    p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='prediction_results.csv')
    return p.parse_args()


def load_test_data(name, mol_blocks, rebuild):
    npz_path = resolve_npz(name, mol_blocks, rebuild=rebuild)
    loader = GraphDataLoader(npz_file=npz_path)
    return DataLoader(loader.pyg_data, batch_size=1, shuffle=False)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader_1 = load_test_data(args.test_data_1, args.mol_blocks, args.rebuild_features)
    test_loader_2 = load_test_data(args.test_data_2, args.mol_blocks, args.rebuild_features)

    model = GCNNet(num_classes=args.num_classes, is_large=args.large).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for b1, b2 in zip(test_loader_1, test_loader_2):
            b1, b2 = b1.to(device), b2.to(device)
            out1 = F.softmax(model(b1.x, b1.edge_index, b1.batch), dim=1)
            out2 = F.softmax(model(b2.x, b2.edge_index, b2.batch), dim=1)
            probs = np.average([out1.cpu().numpy(), out2.cpu().numpy()], axis=0)
            preds = np.argmax(probs, axis=1)
            labels = b1.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    for i in range(args.num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            print(f"Class {i} Accuracy: {np.mean(all_preds[mask] == i):.4f}")

    class_names = ['fail', 'salt', 'cocrystal', 'hydrate/solvate']
    results = pd.DataFrame({
        'True Label': all_labels,
        'Predicted Label': all_preds,
        'Correct': all_preds == all_labels,
        **{f'P({class_names[i]})': all_probs[:, i] for i in range(args.num_classes)},
    })
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

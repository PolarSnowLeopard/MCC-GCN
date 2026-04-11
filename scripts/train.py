"""Pre-training script for MCC-GCN.

Usage:
    python scripts/train.py --data HKU_data_5_total_inbalance --epochs 400
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcc_gcn.utils import seed_everything, resolve_npz
from mcc_gcn.models.gcn import GCNNet, train_epoch, evaluate
from mcc_gcn.models.metrics import calculate_metrics, calculate_detailed_metrics
from mcc_gcn.data.dataset import GraphDataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MCC-GCN Pre-training")
    p.add_argument('--data', type=str, required=True, help='Dataset name (without extension)')
    p.add_argument('--mol-blocks', type=str, default='data/HKU_data.pkl.gz')
    p.add_argument('--rebuild-features', action='store_true', help='Rebuild npz from CSV + pkl.gz')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--save-dir', type=str, default='checkpoints')
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    npz_path = resolve_npz(args.data, args.mol_blocks, rebuild=args.rebuild_features)
    loader = GraphDataLoader(npz_file=npz_path)
    dataset = loader.pyg_data

    # --- Split ---
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=0.1, random_state=args.seed,
    )
    train_loader = DataLoader(
        [dataset[i] for i in train_idx], batch_size=args.batch_size, shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        [dataset[i] for i in val_idx], batch_size=args.batch_size, shuffle=False,
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
    train_labels = [dataset[i].y.item() for i in train_idx]
    val_labels = [dataset[i].y.item() for i in val_idx]
    print(f"Train distribution: {np.bincount(train_labels)}")
    print(f"Val distribution:   {np.bincount(val_labels)}")

    # --- Model ---
    model = GCNNet(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7,
    )

    # --- Training loop ---
    best_val_bacc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_labels, train_preds, train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_labels, val_preds, val_loss = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_loss)

        _, train_bacc, _ = calculate_metrics(train_labels, train_preds, args.num_classes)
        _, val_bacc, _ = calculate_metrics(val_labels, val_preds, args.num_classes)

        msg = (
            f"Epoch {epoch:03d}: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train BACC: {train_bacc:.4f}, Val BACC: {val_bacc:.4f}"
        )

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            msg += f"  * Best"

        print(msg)

    print(f"\nBest Val BACC: {best_val_bacc:.4f}")
    print(f"Model saved to {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()

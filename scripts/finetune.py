"""Fine-tuning script for MCC-GCN.

Usage:
    python scripts/finetune.py --data HKU_data_5_merge_FT_crystal_dataset \
        --pretrained checkpoints/best_model.pth --epochs 200
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
from mcc_gcn.models.metrics import calculate_metrics
from mcc_gcn.data.dataset import GraphDataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MCC-GCN Fine-tuning")
    p.add_argument('--data', type=str, required=True, help='Fine-tuning dataset name')
    p.add_argument('--mol-blocks', type=str, default='data/HKU_data.pkl.gz')
    p.add_argument('--pretrained', type=str, required=True, help='Path to pre-trained model')
    p.add_argument('--rebuild-features', action='store_true')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--large', action='store_true')
    p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--train-layers', type=int, default=1,
                   help='Number of dense layers to unfreeze (1-3, or 0 for all)')
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

    # --- Load pre-trained model and freeze layers ---
    model = GCNNet(num_classes=args.num_classes, is_large=args.large).to(device)
    model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))
    model.ft_setting(train_dense_layer=args.train_layers)
    print(f"Loaded pre-trained weights from {args.pretrained}")
    print(f"Unfreezing {args.train_layers} dense layer(s) for fine-tuning")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable}/{total}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7,
    )

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
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_FT_model.pth'))
            msg += f"  * Best"

        print(msg)

    print(f"\nBest FT Val BACC: {best_val_bacc:.4f}")
    print(f"Model saved to {args.save_dir}/best_FT_model.pth")


if __name__ == '__main__':
    main()

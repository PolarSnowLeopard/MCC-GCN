"""Pre-training script for MCC-GCN.

Usage:
    python scripts/train.py --data HKU_data_5_total_inbalance --epochs 400
"""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcc_gcn.utils import seed_everything, resolve_npz
from mcc_gcn.models.gcn import (
    GCNNet,
    evaluate,
    evaluate_pair_averaged,
    resolve_validation_aggregation,
    train_epoch,
)
from mcc_gcn.models.imbalance import calculate_class_weights
from mcc_gcn.models.metrics import calculate_metrics
from mcc_gcn.models.selection import validation_result_improved
from mcc_gcn.data.dataset import GraphDataLoader
from mcc_gcn.data.splitting import grouped_stratified_split


def parse_args():
    p = argparse.ArgumentParser(description="MCC-GCN Pre-training")
    p.add_argument('--data', type=str, required=True, help='Dataset name (without extension)')
    p.add_argument(
        '--val-data',
        type=str,
        help='Frozen validation NPZ built from a disjoint physical-pair split.',
    )
    p.add_argument('--mol-blocks', type=str, default='data/HKU_data.pkl.gz')
    p.add_argument('--rebuild-features', action='store_true', help='Rebuild npz from CSV + pkl.gz')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--task',
        choices=['binary', 'four-class'],
        default='four-class',
    )
    p.add_argument(
        '--num-classes',
        type=int,
        help='Deprecated explicit override; must agree with --task.',
    )
    p.add_argument(
        '--model-size',
        choices=['small', 'large'],
        default='large',
        help='Historical binary architecture is small; four-class is large.',
    )
    p.add_argument('--validation-fraction', type=float, default=0.1)
    p.add_argument(
        '--class-weighting',
        choices=['none', 'inverse-frequency', 'effective-number'],
        default='effective-number',
        help='Retain all rows and adjust only the loss contribution.',
    )
    p.add_argument(
        '--effective-number-beta',
        type=float,
        default=0.9999,
    )
    p.add_argument(
        '--legacy-row-split',
        action='store_true',
        help=(
            'Allow historical row-level splitting when pair_keys are absent. '
            'This is only for baseline reproduction and can leak augmented pairs.'
        ),
    )
    p.add_argument(
        '--early-stopping-patience',
        type=int,
        default=30,
        help='Epochs without validation BACC improvement; 0 disables stopping.',
    )
    p.add_argument(
        '--tensorboard-dir',
        help='Optional TensorBoard event directory for epoch metrics.',
    )
    p.add_argument('--save-dir', type=str, default='checkpoints')
    return p.parse_args()


def main():
    args = parse_args()
    expected_num_classes = 2 if args.task == 'binary' else 4
    if (
        args.num_classes is not None
        and args.num_classes != expected_num_classes
    ):
        raise ValueError('--num-classes does not agree with --task')
    args.num_classes = expected_num_classes
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    if args.early_stopping_patience < 0:
        raise ValueError('--early-stopping-patience cannot be negative')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    npz_path = resolve_npz(
        args.data,
        args.mol_blocks,
        rebuild=args.rebuild_features,
    )
    label_mode = 'binary' if args.task == 'binary' else 'stored'
    dataset = GraphDataLoader(
        npz_file=npz_path,
        label_mode=label_mode,
    ).pyg_data

    # --- Use a frozen split, or split physical pair groups as a fallback ---
    resolved_val_npz = None
    if args.val_data:
        resolved_val_npz = resolve_npz(
            args.val_data,
            args.mol_blocks,
            rebuild=args.rebuild_features,
        )
        validation_dataset = GraphDataLoader(
            npz_file=resolved_val_npz,
            label_mode=label_mode,
        ).pyg_data
        train_pair_keys = {
            getattr(item, 'pair_key', None) for item in dataset
        }
        validation_pair_keys = {
            getattr(item, 'pair_key', None)
            for item in validation_dataset
        }
        if None in train_pair_keys or None in validation_pair_keys:
            raise ValueError(
                'Frozen train/validation NPZ files must contain pair_keys'
            )
        overlap = train_pair_keys.intersection(validation_pair_keys)
        if overlap:
            raise ValueError(
                f'Frozen train/validation NPZ files overlap on '
                f'{len(overlap)} physical pairs'
            )
        train_items = dataset
        val_items = validation_dataset
        split_mode = 'frozen_disjoint_pair_files'
    else:
        labels = np.asarray(
            [item.y.item() for item in dataset],
            dtype=np.int64,
        )
        pair_keys = [getattr(item, 'pair_key', None) for item in dataset]
        if all(pair_keys):
            grouped_split = grouped_stratified_split(
                labels,
                pair_keys,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
            )
            train_idx = list(grouped_split.train_indices)
            val_idx = list(grouped_split.validation_indices)
            grouped_split.manifest.to_csv(
                os.path.join(args.save_dir, 'split_manifest.csv'),
                index=False,
            )
            split_mode = 'grouped_stratified_pair'
        elif args.legacy_row_split:
            train_idx, val_idx = train_test_split(
                range(len(dataset)),
                test_size=args.validation_fraction,
                random_state=args.seed,
            )
            split_mode = 'legacy_row_split'
            print(
                'WARNING: pair_keys are absent; using leakage-prone legacy '
                'row split'
            )
        else:
            raise ValueError(
                'Feature NPZ does not contain pair_keys. Rebuild corrected '
                'features or pass --legacy-row-split only to reproduce the '
                'historical baseline.'
            )
        train_items = [dataset[i] for i in train_idx]
        val_items = [dataset[i] for i in val_idx]

    drop_last = len(train_items) % args.batch_size == 1
    train_loader = DataLoader(
        train_items, batch_size=args.batch_size, shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_items, batch_size=args.batch_size, shuffle=False,
    )
    validation_aggregation = resolve_validation_aggregation(val_items)

    print(f"Train: {len(train_items)}, Val: {len(val_items)}")
    train_labels = np.asarray(
        [item.y.item() for item in train_items],
        dtype=np.int64,
    )
    val_labels = np.asarray(
        [item.y.item() for item in val_items],
        dtype=np.int64,
    )
    print(f"Train distribution: {np.bincount(train_labels)}")
    print(f"Val distribution:   {np.bincount(val_labels)}")
    print(f"Split mode: {split_mode}")
    print(f"Validation aggregation: {validation_aggregation}")

    # --- Model ---
    model = GCNNet(
        num_classes=args.num_classes,
        model_size=args.model_size,
    ).to(device)
    class_weights = calculate_class_weights(
        train_labels,
        args.num_classes,
        mode=args.class_weighting,
        beta=args.effective_number_beta,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, device=device),
    )
    print(
        f"Class weighting: {args.class_weighting} "
        f"{class_weights.tolist()}"
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7,
    )

    # --- Training loop ---
    run_config = {
        **vars(args),
        'resolved_npz': npz_path,
        'resolved_validation_npz': resolved_val_npz,
        'split_mode': split_mode,
        'validation_aggregation': validation_aggregation,
        'train_rows': len(train_items),
        'validation_rows': len(val_items),
        'class_weights': class_weights.tolist(),
        'device': str(device),
        'torch_version': torch.__version__,
    }
    with open(
        os.path.join(args.save_dir, 'run_config.json'),
        'w',
        encoding='utf-8',
    ) as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
        handle.write('\n')

    summary_writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                'TensorBoard logging requires the tensorboard package. '
                'Install it with: python -m pip install tensorboard'
            ) from exc
        summary_writer = SummaryWriter(
            log_dir=args.tensorboard_dir,
            flush_secs=10,
        )
        summary_writer.add_text(
            'run/config',
            json.dumps(run_config, indent=2, sort_keys=True),
            global_step=0,
        )
        print(f'TensorBoard logs: {args.tensorboard_dir}')

    best_val_bacc = -1.0
    best_val_loss = float('inf')
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_train_labels, train_preds, train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        if validation_aggregation == 'pair_probability_mean':
            epoch_val_labels, val_preds, val_loss = evaluate_pair_averaged(
                model, val_loader, criterion, device,
            )
        else:
            epoch_val_labels, val_preds, val_loss = evaluate(
                model, val_loader, criterion, device,
            )
        scheduler.step(val_loss)

        _, train_bacc, _ = calculate_metrics(
            epoch_train_labels,
            train_preds,
            args.num_classes,
        )
        _, val_bacc, _ = calculate_metrics(
            epoch_val_labels,
            val_preds,
            args.num_classes,
        )
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'validation_loss': val_loss,
                'train_balanced_accuracy': train_bacc,
                'validation_balanced_accuracy': val_bacc,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
        )
        if summary_writer is not None:
            summary_writer.add_scalar('loss/train', train_loss, epoch)
            summary_writer.add_scalar('loss/validation', val_loss, epoch)
            summary_writer.add_scalar(
                'balanced_accuracy/train',
                train_bacc,
                epoch,
            )
            summary_writer.add_scalar(
                'balanced_accuracy/validation',
                val_bacc,
                epoch,
            )
            summary_writer.add_scalar(
                'optimization/learning_rate',
                optimizer.param_groups[0]['lr'],
                epoch,
            )

        msg = (
            f"Epoch {epoch:03d}: "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train BACC: {train_bacc:.4f}, Val BACC: {val_bacc:.4f}"
        )

        if validation_result_improved(
            val_bacc,
            val_loss,
            best_val_bacc,
            best_val_loss,
        ):
            best_val_bacc = val_bacc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            msg += "  * Best"
        else:
            epochs_without_improvement += 1

        print(msg)
        if (
            args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f'Early stopping after epoch {epoch}')
            break

    torch.save(
        model.state_dict(),
        os.path.join(args.save_dir, 'last_model.pth'),
    )
    with open(
        os.path.join(args.save_dir, 'training_history.csv'),
        'w',
        newline='',
        encoding='utf-8',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    selection = {
        'best_epoch': best_epoch,
        'best_validation_balanced_accuracy': best_val_bacc,
        'best_validation_loss': best_val_loss,
        'epochs_completed': len(history),
    }
    with open(
        os.path.join(args.save_dir, 'selection_result.json'),
        'w',
        encoding='utf-8',
    ) as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
        handle.write('\n')

    if summary_writer is not None:
        summary_writer.add_scalar(
            'selection/best_validation_balanced_accuracy',
            best_val_bacc,
            best_epoch,
        )
        summary_writer.add_scalar(
            'selection/best_validation_loss',
            best_val_loss,
            best_epoch,
        )
        summary_writer.flush()
        summary_writer.close()

    print(f"\nBest Val BACC: {best_val_bacc:.4f} at epoch {best_epoch}")
    print(f"Model saved to {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()

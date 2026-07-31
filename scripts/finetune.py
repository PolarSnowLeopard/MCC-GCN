"""Leakage-resistant fine-tuning for MCC-GCN.

Selection mode uses either a disjoint validation NPZ or a grouped split of the
fine-tuning pairs. Final-fit mode retrains on every fine-tuning pair for a
fixed number of epochs selected by a previous selection run.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcc_gcn.data.dataset import GraphDataLoader
from mcc_gcn.data.splitting import grouped_stratified_split
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
from mcc_gcn.utils import resolve_npz, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="MCC-GCN fine-tuning")
    parser.add_argument(
        "--data",
        required=True,
        help="Fine-tuning NPZ, including pair_keys in corrected runs.",
    )
    parser.add_argument(
        "--val-data",
        help=(
            "Optional frozen validation NPZ disjoint from training and final "
            "holdout data. If omitted, selection mode makes a grouped split."
        ),
    )
    parser.add_argument(
        "--holdout-data",
        action="append",
        default=[],
        help=(
            "NPZ reserved for final evaluation. May be repeated. It is loaded "
            "only to verify that no pair enters training or validation."
        ),
    )
    parser.add_argument("--mol-blocks", default="data/HKU_data.pkl.gz")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["select", "final-fit"],
        default="select",
        help=(
            "select chooses an epoch using validation; final-fit uses all "
            "fine-tuning rows for exactly --epochs epochs."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--task",
        choices=["binary", "four-class"],
        default="four-class",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Deprecated explicit override; must agree with --task.",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "large"],
        default="large",
        help="Must match the pretraining checkpoint architecture.",
    )
    parser.add_argument(
        "--train-layers",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Dense layers to unfreeze; 0 unfreezes the complete network.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse-frequency", "effective-number"],
        default="effective-number",
    )
    parser.add_argument(
        "--manual-class-weights",
        help=(
            "Comma-separated override, for example 1,1,1,2. Intended only "
            "for a documented compatibility or ablation run."
        ),
    )
    parser.add_argument("--effective-number-beta", type=float, default=0.9999)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Selection epochs without BACC improvement; 0 disables stopping.",
    )
    parser.add_argument(
        "--legacy-row-split",
        action="store_true",
        help=(
            "Allow historical row-level splitting when pair_keys are absent. "
            "This can leak A/B augmented copies and is baseline-only."
        ),
    )
    parser.add_argument(
        "--tensorboard-dir",
        help="Optional TensorBoard event directory for epoch metrics.",
    )
    parser.add_argument("--save-dir", default="checkpoints")
    return parser.parse_args()


def _load_items(path, args):
    resolved = resolve_npz(
        path,
        args.mol_blocks,
        rebuild=args.rebuild_features,
    )
    label_mode = "binary" if args.task == "binary" else "stored"
    return resolved, GraphDataLoader(
        npz_file=resolved,
        label_mode=label_mode,
    ).pyg_data


def _pair_keys(items, description):
    keys = [getattr(item, "pair_key", None) for item in items]
    if not all(keys):
        raise ValueError(
            f"{description} is missing pair_keys; rebuild corrected features"
        )
    return set(keys)


def _split_selection_data(dataset, args):
    labels = np.asarray([item.y.item() for item in dataset], dtype=np.int64)
    pair_keys = [getattr(item, "pair_key", None) for item in dataset]
    if all(pair_keys):
        split = grouped_stratified_split(
            labels,
            pair_keys,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        split.manifest.to_csv(
            os.path.join(args.save_dir, "split_manifest.csv"),
            index=False,
        )
        return (
            [dataset[index] for index in split.train_indices],
            [dataset[index] for index in split.validation_indices],
            "grouped_stratified_pair",
        )
    if not args.legacy_row_split:
        raise ValueError(
            "Fine-tuning NPZ does not contain pair_keys. Rebuild corrected "
            "features, or use --legacy-row-split only for baseline reproduction."
        )
    train_indices, validation_indices = train_test_split(
        range(len(dataset)),
        test_size=args.validation_fraction,
        random_state=args.seed,
    )
    print(
        "WARNING: pair_keys are absent; using leakage-prone legacy row split"
    )
    return (
        [dataset[index] for index in train_indices],
        [dataset[index] for index in validation_indices],
        "legacy_row_split",
    )


def _write_history(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    expected_num_classes = 2 if args.task == "binary" else 4
    if (
        args.num_classes is not None
        and args.num_classes != expected_num_classes
    ):
        raise ValueError("--num-classes does not agree with --task")
    args.num_classes = expected_num_classes
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.mode == "final-fit" and args.val_data:
        raise ValueError("--val-data is not used in final-fit mode")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_npz, dataset = _load_items(args.data, args)
    if not all(getattr(item, "pair_key", None) for item in dataset):
        if not args.legacy_row_split:
            raise ValueError(
                "Fine-tuning NPZ does not contain pair_keys. Rebuild corrected "
                "features, or use --legacy-row-split only for baseline "
                "reproduction."
            )
        print(
            "WARNING: fine-tuning data has no pair_keys; pair-level leakage "
            "checks are unavailable"
        )
    resolved_val_npz = None
    if args.mode == "final-fit":
        train_items = dataset
        validation_items = []
        split_mode = "final_fit_all_pairs"
    elif args.val_data:
        resolved_val_npz, validation_items = _load_items(args.val_data, args)
        if args.legacy_row_split and (
            not all(getattr(item, "pair_key", None) for item in dataset)
            or not all(
                getattr(item, "pair_key", None)
                for item in validation_items
            )
        ):
            print(
                "WARNING: legacy validation data has no pair_keys; overlap "
                "and holdout-leakage checks are unavailable"
            )
        else:
            train_keys = _pair_keys(dataset, "Fine-tuning data")
            validation_keys = _pair_keys(validation_items, "Validation data")
            overlap = train_keys.intersection(validation_keys)
            if overlap:
                raise ValueError(
                    "Fine-tuning and validation NPZ files overlap on "
                    f"{len(overlap)} physical pairs"
                )
        train_items = dataset
        split_mode = "frozen_disjoint_pair_files"
    else:
        train_items, validation_items, split_mode = _split_selection_data(
            dataset,
            args,
        )

    protected_keys = set()
    resolved_holdout_npz = []
    for holdout_path in args.holdout_data:
        resolved_path, holdout_items = _load_items(holdout_path, args)
        resolved_holdout_npz.append(resolved_path)
        holdout_keys = _pair_keys(holdout_items, f"Holdout data {resolved_path}")
        overlap = holdout_keys.intersection(
            _pair_keys(dataset, "Fine-tuning data")
        )
        if overlap:
            raise ValueError(
                f"Holdout {resolved_path} overlaps fine-tuning data on "
                f"{len(overlap)} physical pairs"
            )
        if validation_items:
            overlap = holdout_keys.intersection(
                _pair_keys(validation_items, "Validation data")
            )
            if overlap:
                raise ValueError(
                    f"Holdout {resolved_path} overlaps validation data on "
                    f"{len(overlap)} physical pairs"
                )
        protected_keys.update(holdout_keys)

    drop_last = (
        args.train_layers != 1
        and len(train_items) % args.batch_size == 1
    )
    train_loader = DataLoader(
        train_items,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    validation_loader = (
        DataLoader(validation_items, batch_size=256, shuffle=False)
        if validation_items
        else None
    )
    validation_aggregation = resolve_validation_aggregation(validation_items)

    train_labels = np.asarray(
        [item.y.item() for item in train_items],
        dtype=np.int64,
    )
    if args.manual_class_weights:
        class_weights = np.asarray(
            [
                float(value)
                for value in args.manual_class_weights.split(",")
            ],
            dtype=np.float32,
        )
        if (
            len(class_weights) != args.num_classes
            or np.any(class_weights <= 0)
        ):
            raise ValueError(
                "--manual-class-weights must provide one positive value "
                "per class"
            )
        resolved_weighting = "manual"
    else:
        class_weights = calculate_class_weights(
            train_labels,
            args.num_classes,
            mode=args.class_weighting,
            beta=args.effective_number_beta,
        )
        resolved_weighting = args.class_weighting

    model = GCNNet(
        num_classes=args.num_classes,
        model_size=args.model_size,
    ).to(device)
    state_dict = torch.load(
        args.pretrained,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.ft_setting(train_dense_layer=args.train_layers)

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded pre-trained weights from {args.pretrained}")
    print(f"Mode: {args.mode}; split: {split_mode}")
    print(
        f"Train: {len(train_items)}, "
        f"Validation: {len(validation_items)}, "
        f"protected holdout pairs: {len(protected_keys)}"
    )
    print(
        f"Unfreezing setting {args.train_layers}; "
        f"trainable params: {trainable}/{total}"
    )
    print(
        f"Class weighting: {resolved_weighting} "
        f"{class_weights.tolist()}"
    )
    if validation_loader is not None:
        print(f"Validation aggregation: {validation_aggregation}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, device=device),
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        if validation_loader is not None
        else None
    )

    run_config = {
        **vars(args),
        "resolved_train_npz": train_npz,
        "resolved_validation_npz": resolved_val_npz,
        "resolved_holdout_npz": resolved_holdout_npz,
        "split_mode": split_mode,
        "validation_aggregation": validation_aggregation,
        "train_rows": len(train_items),
        "validation_rows": len(validation_items),
        "class_weights": class_weights.tolist(),
        "resolved_class_weighting": resolved_weighting,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    with open(
        os.path.join(args.save_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary_writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requires the tensorboard package. "
                "Install it with: python -m pip install tensorboard"
            ) from exc
        summary_writer = SummaryWriter(
            log_dir=args.tensorboard_dir,
            flush_secs=10,
        )
        summary_writer.add_text(
            "run/config",
            json.dumps(run_config, indent=2, sort_keys=True),
            global_step=0,
        )
        print(f"TensorBoard logs: {args.tensorboard_dir}")

    history = []
    best_validation_bacc = -1.0
    best_validation_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        epoch_train_labels, train_predictions, train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        _, train_bacc, _ = calculate_metrics(
            epoch_train_labels,
            train_predictions,
            args.num_classes,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_balanced_accuracy": train_bacc,
            "validation_loss": "",
            "validation_balanced_accuracy": "",
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        message = (
            f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, "
            f"Train BACC: {train_bacc:.4f}"
        )

        if validation_loader is not None:
            if validation_aggregation == "pair_probability_mean":
                (
                    validation_labels,
                    validation_predictions,
                    validation_loss,
                ) = evaluate_pair_averaged(
                    model,
                    validation_loader,
                    criterion,
                    device,
                )
            else:
                (
                    validation_labels,
                    validation_predictions,
                    validation_loss,
                ) = evaluate(
                    model,
                    validation_loader,
                    criterion,
                    device,
                )
            _, validation_bacc, _ = calculate_metrics(
                validation_labels,
                validation_predictions,
                args.num_classes,
            )
            scheduler.step(validation_loss)
            row["validation_loss"] = validation_loss
            row["validation_balanced_accuracy"] = validation_bacc
            message += (
                f", Val Loss: {validation_loss:.4f}, "
                f"Val BACC: {validation_bacc:.4f}"
            )
            if validation_result_improved(
                validation_bacc,
                validation_loss,
                best_validation_bacc,
                best_validation_loss,
            ):
                best_validation_bacc = validation_bacc
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, "best_FT_model.pth"),
                )
                message += "  * Best"
            else:
                epochs_without_improvement += 1
        history.append(row)
        if summary_writer is not None:
            summary_writer.add_scalar("loss/train", train_loss, epoch)
            summary_writer.add_scalar(
                "balanced_accuracy/train",
                train_bacc,
                epoch,
            )
            summary_writer.add_scalar(
                "optimization/learning_rate",
                optimizer.param_groups[0]["lr"],
                epoch,
            )
            if validation_loader is not None:
                summary_writer.add_scalar(
                    "loss/validation",
                    validation_loss,
                    epoch,
                )
                summary_writer.add_scalar(
                    "balanced_accuracy/validation",
                    validation_bacc,
                    epoch,
                )
        print(message)

        if (
            validation_loader is not None
            and args.early_stopping_patience
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"Early stopping after epoch {epoch}")
            break

    last_path = os.path.join(args.save_dir, "last_FT_model.pth")
    torch.save(model.state_dict(), last_path)
    _write_history(
        os.path.join(args.save_dir, "training_history.csv"),
        history,
    )

    if args.mode == "final-fit":
        final_path = os.path.join(args.save_dir, "final_FT_model.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Final-fit model saved to {final_path}")
    else:
        selection = {
            "best_epoch": best_epoch,
            "best_validation_balanced_accuracy": best_validation_bacc,
            "best_validation_loss": best_validation_loss,
            "epochs_completed": len(history),
        }
        with open(
            os.path.join(args.save_dir, "selection_result.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(selection, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(
            f"Best validation BACC: {best_validation_bacc:.4f} "
            f"at epoch {best_epoch}"
        )
        print(
            "Use that epoch count in a fresh --mode final-fit run before "
            "evaluating the protected holdout."
        )
    if summary_writer is not None:
        if args.mode == "select":
            summary_writer.add_scalar(
                "selection/best_validation_balanced_accuracy",
                best_validation_bacc,
                best_epoch,
            )
            summary_writer.add_scalar(
                "selection/best_validation_loss",
                best_validation_loss,
                best_epoch,
            )
        summary_writer.flush()
        summary_writer.close()


if __name__ == "__main__":
    main()

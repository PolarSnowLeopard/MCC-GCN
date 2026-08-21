#!/usr/bin/env python3
"""Fine-tune only on Minoxidil and evaluate the protected KPX/KPR-64 set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcc_gcn.data.minoxidil_only import load_minoxidil_only_data
from mcc_gcn.models.gcn import GCNNet, train_epoch
from mcc_gcn.models.metrics import calculate_detailed_metrics, calculate_metrics
from mcc_gcn.utils import seed_everything

CLASS_NAMES = ["negative", "salt", "cocrystal", "hydrate_or_solvate"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument(
        "--fine-tune-data",
        default="data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz",
    )
    parser.add_argument(
        "--holdout-data-1",
        default="data/HKU_data_6_experiment_1.npz",
    )
    parser.add_argument(
        "--holdout-data-2",
        default="data/HKU_data_6_experiment_2.npz",
    )
    parser.add_argument(
        "--fine-tune-manifest",
        default="data/manifests/finetune-34-lock-v1.csv",
    )
    parser.add_argument(
        "--external-manifest",
        default="data/manifests/external-64-split-v1.csv",
    )
    parser.add_argument(
        "--input-mode",
        choices=["legacy-padded", "trimmed"],
        default="legacy-padded",
        help=(
            "legacy-padded reproduces submitted dense features; trimmed is "
            "a clearly labeled loader-correction sensitivity analysis"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-1)
    parser.add_argument("--train-layers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--class-weights",
        default="1,1,1,2",
        help="Executed four-class fine-tuning weights, in label order",
    )
    parser.add_argument(
        "--seeds",
        default="12,42,43",
        help="Comma-separated fine-tuning seeds; seed 12 is the submitted run",
    )
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values")
    return seeds


def _parse_class_weights(value):
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != 4 or any(weight <= 0 for weight in weights):
        raise ValueError("--class-weights must contain four positive values")
    return weights


def _infer(model, items, device, batch_size):
    probabilities = []
    labels = []
    pair_keys = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            probabilities.append(
                F.softmax(model(batch.x, batch.edge_index, batch.batch), dim=1)
                .cpu()
                .numpy()
            )
            labels.extend(batch.y.cpu().numpy().tolist())
            keys = batch.pair_key
            pair_keys.extend([keys] if isinstance(keys, str) else list(keys))
    return np.concatenate(probabilities), np.asarray(labels), pair_keys


def _evaluate_external(model, data, device, batch_size):
    probabilities_ab, labels_ab, keys_ab = _infer(
        model, data.external_ab_items, device, batch_size
    )
    probabilities_ba, labels_ba, keys_ba = _infer(
        model, data.external_ba_items, device, batch_size
    )
    if not np.array_equal(labels_ab, labels_ba):
        raise ValueError("KPX/KPR-64 orientation labels differ")
    if keys_ab != keys_ba:
        raise ValueError("KPX/KPR-64 orientation pair keys differ")
    probabilities = 0.5 * (probabilities_ab + probabilities_ba)
    predictions = probabilities.argmax(axis=1)
    per_class, balanced_accuracy, accuracy = calculate_metrics(
        labels_ab, predictions, 4
    )
    detailed = calculate_detailed_metrics(labels_ab, predictions, 4)
    total_variation = 0.5 * np.abs(probabilities_ab - probabilities_ba).sum(axis=1)
    predictions_ab = probabilities_ab.argmax(axis=1)
    predictions_ba = probabilities_ba.argmax(axis=1)
    return {
        "labels": labels_ab,
        "predictions": predictions,
        "probabilities": probabilities,
        "probabilities_ab": probabilities_ab,
        "probabilities_ba": probabilities_ba,
        "pair_keys": keys_ab,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion_matrix(
            labels_ab, predictions, labels=list(range(4))
        ),
        "detailed_metrics": detailed,
        "orientation_agreement": float(np.mean(predictions_ab == predictions_ba)),
        "orientation_total_variation": total_variation,
    }


def _json_metrics(result):
    detailed = result["detailed_metrics"]
    return {
        "overall_accuracy": result["accuracy"],
        "balanced_accuracy": result["balanced_accuracy"],
        "per_class_accuracy": {
            name: float(result["per_class_accuracy"][index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": result["confusion_matrix"].tolist(),
        "macro_precision": float(detailed["macro_precision"]),
        "macro_recall": float(detailed["macro_recall"]),
        "macro_f1": float(detailed["macro_f1"]),
        "orientation_agreement": result["orientation_agreement"],
        "mean_orientation_total_variation": float(
            result["orientation_total_variation"].mean()
        ),
        "max_orientation_total_variation": float(
            result["orientation_total_variation"].max()
        ),
    }


def _write_evaluation(result, external_table, output):
    table = external_table.copy()
    table["true_label"] = result["labels"]
    table["predicted_label"] = result["predictions"]
    table["correct"] = result["labels"] == result["predictions"]
    table["orientation_total_variation"] = result["orientation_total_variation"]
    for index, name in enumerate(CLASS_NAMES):
        table[f"p_{name}"] = result["probabilities"][:, index]
        table[f"p_ab_{name}"] = result["probabilities_ab"][:, index]
        table[f"p_ba_{name}"] = result["probabilities_ba"][:, index]
    table.to_csv(output, index=False)
    return _json_metrics(result)


def _tensorboard_writer(path, config):
    if path is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard logging requires tensorboard") from exc
    writer = SummaryWriter(log_dir=str(path), flush_secs=10)
    writer.add_text("run/config", json.dumps(config, indent=2, sort_keys=True), 0)
    return writer


def _train_seed(args, data, seed, device, output_dir, base_config, class_weights):
    seed_everything(seed)
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model = GCNNet(num_classes=4, model_size="large").to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.ft_setting(train_dense_layer=args.train_layers)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )
    loader = DataLoader(
        data.minoxidil_items,
        batch_size=args.batch_size,
        shuffle=True,
    )
    run_config = {
        **base_config,
        "seed": seed,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (seed_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tensorboard_path = (
        Path(args.tensorboard_dir) / args.input_mode / f"seed-{seed}"
        if args.tensorboard_dir
        else None
    )
    writer = _tensorboard_writer(tensorboard_path, run_config)
    history = []
    for epoch in range(1, args.epochs + 1):
        labels, predictions, loss = train_epoch(
            model, loader, optimizer, criterion, device
        )
        _, balanced_accuracy, accuracy = calculate_metrics(labels, predictions, 4)
        scheduler.step(loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_balanced_accuracy": balanced_accuracy,
                "learning_rate": learning_rate,
            }
        )
        if writer is not None:
            writer.add_scalar("loss/train", loss, epoch)
            writer.add_scalar("accuracy/train", accuracy, epoch)
            writer.add_scalar("balanced_accuracy/train", balanced_accuracy, epoch)
            writer.add_scalar("optimization/learning_rate", learning_rate, epoch)
        print(
            f"seed {seed} epoch {epoch:03d}: loss={loss:.4f}, "
            f"accuracy={accuracy:.4f}, BACC={balanced_accuracy:.4f}"
        )
    if writer is not None:
        writer.close()

    checkpoint = seed_dir / "minoxidil_only_FT_model.pth"
    torch.save(model.state_dict(), checkpoint)
    with open(
        seed_dir / "training_history.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(history)
    result = _evaluate_external(model, data, device, args.batch_size)
    prediction_path = seed_dir / "kpxkpr64_predictions.csv"
    metrics = _write_evaluation(result, data.external_pairs, prediction_path)
    metrics.update(
        {
            "schema_version": "mcc-gcn-minoxidil-only-evaluation-v1",
            "seed": seed,
            "input_mode": args.input_mode,
            "fine_tuning_physical_pairs": 20,
            "fine_tuning_ordered_rows": 40,
            "external_physical_pairs": 64,
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "predictions": {
                "path": str(prediction_path),
                "sha256": _sha256(prediction_path),
            },
        }
    )
    (seed_dir / "kpxkpr64_predictions.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"seed {seed} KPXKPR-64: accuracy={metrics['overall_accuracy']:.4f}, "
        f"BACC={metrics['balanced_accuracy']:.4f}"
    )
    return metrics


def _aggregate(metrics):
    scalar_names = [
        "overall_accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "orientation_agreement",
        "mean_orientation_total_variation",
    ]
    aggregate = {}
    for name in scalar_names:
        values = np.asarray([row[name] for row in metrics], dtype=float)
        aggregate[name] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1))
            if len(values) > 1
            else 0.0,
            "values": values.tolist(),
        }
    for class_name in CLASS_NAMES:
        values = np.asarray(
            [row["per_class_accuracy"][class_name] for row in metrics], dtype=float
        )
        aggregate[f"{class_name}_accuracy"] = {
            "mean": float(values.mean()),
            "sample_standard_deviation": float(values.std(ddof=1))
            if len(values) > 1
            else 0.0,
            "values": values.tolist(),
        }
    return aggregate


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    seeds = _parse_seeds(args.seeds)
    class_weights = _parse_class_weights(args.class_weights)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve_padding = args.input_mode == "legacy-padded"
    data = load_minoxidil_only_data(
        fine_tune_npz=args.fine_tune_data,
        holdout_ab_npz=args.holdout_data_1,
        holdout_ba_npz=args.holdout_data_2,
        fine_tune_manifest=args.fine_tune_manifest,
        external_manifest=args.external_manifest,
        preserve_padding=preserve_padding,
    )
    minoxidil_labels = [int(item.y.item()) for item in data.minoxidil_items[:20]]
    external_labels = [int(item.y.item()) for item in data.external_ab_items]
    base_config = {
        "schema_version": "mcc-gcn-minoxidil-only-run-v1",
        "task": "four-class",
        "model_size": "large",
        "input_mode": args.input_mode,
        "padding_preserved": preserve_padding,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "train_layers": args.train_layers,
        "class_weights": class_weights,
        "scheduler_monitor": "fine_tuning_train_loss",
        "target_labels_used_for_training_or_model_selection": False,
        "fine_tuning_physical_pairs": 20,
        "fine_tuning_ordered_rows": 40,
        "fine_tuning_class_counts": {
            CLASS_NAMES[index]: int(
                np.count_nonzero(np.asarray(minoxidil_labels) == index)
            )
            for index in range(4)
        },
        "external_physical_pairs": 64,
        "external_class_counts": {
            CLASS_NAMES[index]: int(
                np.count_nonzero(np.asarray(external_labels) == index)
            )
            for index in range(4)
        },
        "data_leakage_check": {
            "target_pairs_in_fine_tuning": 0,
            "physical_pair_membership_locked_before_orientation_augmentation": True,
        },
        "source_artifacts": {
            name: {"path": path, "sha256": _sha256(path)}
            for name, path in {
                "pretrained_checkpoint": args.checkpoint,
                "fine_tune_features": args.fine_tune_data,
                "holdout_ab_features": args.holdout_data_1,
                "holdout_ba_features": args.holdout_data_2,
                "fine_tune_manifest": args.fine_tune_manifest,
                "external_manifest": args.external_manifest,
            }.items()
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(base_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = GCNNet(num_classes=4, model_size="large").to(device)
    pretrained.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    zero_shot_result = _evaluate_external(pretrained, data, device, args.batch_size)
    zero_shot_path = output_dir / "pretrained_kpxkpr64_predictions.csv"
    zero_shot_metrics = _write_evaluation(
        zero_shot_result, data.external_pairs, zero_shot_path
    )
    zero_shot_metrics.update(
        {
            "schema_version": "mcc-gcn-minoxidil-only-evaluation-v1",
            "stage": "pretrained_without_fine_tuning",
            "input_mode": args.input_mode,
            "predictions": {
                "path": str(zero_shot_path),
                "sha256": _sha256(zero_shot_path),
            },
        }
    )
    (output_dir / "pretrained_kpxkpr64_predictions.metrics.json").write_text(
        json.dumps(zero_shot_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pretrained KPXKPR-64: "
        f"accuracy={zero_shot_metrics['overall_accuracy']:.4f}, "
        f"BACC={zero_shot_metrics['balanced_accuracy']:.4f}"
    )

    fine_tuned_metrics = [
        _train_seed(
            args,
            data,
            seed,
            device,
            output_dir,
            base_config,
            class_weights,
        )
        for seed in seeds
    ]
    summary = {
        "schema_version": "mcc-gcn-minoxidil-only-summary-v1",
        "input_mode": args.input_mode,
        "seeds": seeds,
        "pretrained": zero_shot_metrics,
        "minoxidil_only_fine_tuned": _aggregate(fine_tuned_metrics),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["minoxidil_only_fine_tuned"], indent=2, sort_keys=True))
    print(f"Minoxidil-only experiment complete: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a dual-branch SMILES CNN on the frozen three-API experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.splitting import grouped_stratified_split
from mcc_gcn.models.imbalance import calculate_class_weights
from mcc_gcn.models.smiles_cnn import (
    DualSmilesCNN,
    SmilesPairEvaluationDataset,
    SmilesPairTrainingDataset,
    SmilesTokenizer,
    task_labels,
)
from mcc_gcn.utils import seed_everything


CLASS_NAMES = {
    "binary": ["negative", "positive"],
    "four-class": [
        "negative",
        "salt",
        "cocrystal",
        "hydrate_or_solvate",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/revision-three-api",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="binary,four-class")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--smiles-variants", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--pretrain-patience", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=15)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--effective-number-beta", type=float, default=0.9999)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--tensorboard-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pretrain-limit-per-class",
        type=int,
        default=0,
        help="Balanced development-only source subset; zero uses all rows.",
    )
    return parser.parse_args()


def _parse_list(value: str, conversion=str):
    return [
        conversion(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def _load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, keep_default_na=False)
    required = {
        "reactant_A",
        "reactant_B",
        "label_int",
        "pair_key",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if table["pair_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate physical pairs")
    return table


def _development_subset(table, task, per_class, seed):
    if per_class <= 0:
        return table
    labels = task_labels(table, task)
    parts = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) < per_class:
            raise ValueError(
                f"Class {label} has fewer than {per_class} source pairs"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        parts.append(table.iloc[rng.choice(indices, per_class, replace=False)])
    return pd.concat(parts, ignore_index=True).sort_values("pair_key").reset_index(
        drop=True
    )


def _model(tokenizer, task):
    return DualSmilesCNN(
        len(tokenizer.vocabulary),
        len(CLASS_NAMES[task]),
    )


def _clone_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _write_history(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(model, table, tokenizer, task, batch_size, device):
    dataset = SmilesPairEvaluationDataset(table, tokenizer, task=task)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probabilities = []
    all_forward = []
    all_reverse = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for left, right, reverse_left, reverse_right, labels in loader:
            left = left.to(device)
            right = right.to(device)
            reverse_left = reverse_left.to(device)
            reverse_right = reverse_right.to(device)
            forward = torch.softmax(model(left, right), dim=1)
            reverse = torch.softmax(model(reverse_left, reverse_right), dim=1)
            averaged = (forward + reverse) / 2.0
            all_forward.append(forward.cpu())
            all_reverse.append(reverse.cpu())
            all_probabilities.append(averaged.cpu())
            all_labels.append(labels)
    probabilities = torch.cat(all_probabilities).numpy()
    forward = torch.cat(all_forward).numpy()
    reverse = torch.cat(all_reverse).numpy()
    labels = torch.cat(all_labels).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "forward_probabilities": forward,
        "reverse_probabilities": reverse,
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "orientation_agreement": float(
            np.mean(forward.argmax(axis=1) == reverse.argmax(axis=1))
        ),
        "mean_orientation_total_variation": float(
            np.mean(np.abs(forward - reverse).sum(axis=1) / 2.0)
        ),
    }


def _fit(
    model,
    train_table,
    validation_table,
    tokenizer,
    *,
    task,
    epochs,
    patience,
    batch_size,
    lr,
    weight_decay,
    beta,
    variants,
    seed,
    device,
    output_dir,
    tensorboard_dir=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = SmilesPairTrainingDataset(
        train_table,
        tokenizer,
        task=task,
        variants=variants,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    class_weights = calculate_class_weights(
        task_labels(train_table, task),
        len(CLASS_NAMES[task]),
        mode="effective-number",
        beta=beta,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device)
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    writer = None
    if tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tensorboard_dir)

    best_state = None
    best_epoch = 0
    best_balanced_accuracy = float("-inf")
    best_accuracy = float("-inf")
    stale_epochs = 0
    history = []
    try:
        for epoch in range(1, epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            labels_seen = []
            predictions_seen = []
            for left, right, labels in loader:
                left = left.to(device)
                right = right.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(left, right)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                labels_seen.extend(labels.detach().cpu().tolist())
                predictions_seen.extend(logits.argmax(dim=1).detach().cpu().tolist())
            train_loss = total_loss / len(train_dataset)
            train_bacc = float(
                balanced_accuracy_score(labels_seen, predictions_seen)
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_balanced_accuracy": train_bacc,
                "validation_accuracy": "",
                "validation_balanced_accuracy": "",
            }
            marker = ""
            if validation_table is not None:
                validation = _evaluate(
                    model,
                    validation_table,
                    tokenizer,
                    task,
                    batch_size,
                    device,
                )
                val_bacc = validation["balanced_accuracy"]
                val_accuracy = validation["accuracy"]
                row["validation_accuracy"] = val_accuracy
                row["validation_balanced_accuracy"] = val_bacc
                improved = (
                    val_bacc > best_balanced_accuracy + 1e-12
                    or (
                        abs(val_bacc - best_balanced_accuracy) <= 1e-12
                        and val_accuracy > best_accuracy + 1e-12
                    )
                )
                if improved:
                    best_balanced_accuracy = val_bacc
                    best_accuracy = val_accuracy
                    best_epoch = epoch
                    best_state = _clone_state(model)
                    stale_epochs = 0
                    marker = " * Best"
                else:
                    stale_epochs += 1
            history.append(row)
            print(
                f"Epoch {epoch:03d}: loss={train_loss:.4f}, "
                f"train BACC={train_bacc:.4f}"
                + (
                    f", val BACC={row['validation_balanced_accuracy']:.4f}"
                    if validation_table is not None
                    else ""
                )
                + marker,
                flush=True,
            )
            if writer:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("balanced_accuracy/train", train_bacc, epoch)
                if validation_table is not None:
                    writer.add_scalar(
                        "balanced_accuracy/validation",
                        row["validation_balanced_accuracy"],
                        epoch,
                    )
            if validation_table is not None and patience > 0:
                if stale_epochs >= patience:
                    print(f"Early stopping after epoch {epoch}")
                    break
    finally:
        if writer:
            writer.close()

    if validation_table is None:
        best_state = _clone_state(model)
        best_epoch = epochs
    elif best_state is None:
        raise AssertionError("Validation did not produce a best model")
    model.load_state_dict(best_state)
    checkpoint = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint)
    _write_history(output_dir / "training_history.csv", history)
    result = {
        "schema_version": "mcc-gcn-smiles-cnn-selection-v1",
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": (
            best_balanced_accuracy if validation_table is not None else None
        ),
        "best_validation_accuracy": (
            best_accuracy if validation_table is not None else None
        ),
        "epochs_completed": len(history),
        "train_physical_pairs": len(train_table),
        "validation_physical_pairs": (
            len(validation_table) if validation_table is not None else 0
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "class_weights": class_weights.tolist(),
    }
    (output_dir / "selection_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _write_predictions(path, table, evaluation, task):
    class_names = CLASS_NAMES[task]
    pair_key_column = (
        "representative_pair_key"
        if "representative_pair_key" in table.columns
        else "pair_key"
    )
    forward_predictions = evaluation["forward_probabilities"].argmax(axis=1)
    reverse_predictions = evaluation["reverse_probabilities"].argmax(axis=1)
    values = {
        "Pair Key": table[pair_key_column].to_numpy(),
        "True Label": evaluation["labels"],
        "Predicted Label": evaluation["predictions"],
        "Correct": evaluation["labels"] == evaluation["predictions"],
        "Predicted Label (A_B)": forward_predictions,
        "Predicted Label (B_A)": reverse_predictions,
        "Orientation Agreement": forward_predictions == reverse_predictions,
    }
    for index, name in enumerate(class_names):
        values[f"P({name})"] = evaluation["probabilities"][:, index]
        values[f"P(A_B,{name})"] = evaluation["forward_probabilities"][:, index]
        values[f"P(B_A,{name})"] = evaluation["reverse_probabilities"][:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)
    metrics = {
        "schema_version": "mcc-gcn-smiles-cnn-evaluation-v1",
        "task": task,
        "physical_pairs": len(table),
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "orientation_agreement": evaluation["orientation_agreement"],
        "mean_orientation_total_variation": evaluation[
            "mean_orientation_total_variation"
        ],
        "confusion_matrix": confusion_matrix(
            evaluation["labels"],
            evaluation["predictions"],
            labels=list(range(len(class_names))),
        ).tolist(),
    }
    path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_prediction_summary(
    root,
    task,
    target_table,
    predictions,
    output,
    seed,
    *,
    allow_incomplete=False,
):
    command = [
        sys.executable,
        str(root / "scripts" / "summarize_target_api_predictions.py"),
        "--task",
        task,
        "--target-table",
        str(target_table),
    ]
    for path in predictions:
        command.extend(["--prediction", str(path)])
    command.extend(
        [
            "--seed",
            str(seed),
            "--output-dir",
            str(output),
        ]
    )
    if allow_incomplete:
        command.append("--allow-incomplete")
    subprocess.run(command, check=True)


def _load_state(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def _run_task_seed(args, root, data_root, output_root, task, seed, device):
    task_root = output_root / task / f"seed-{seed}"
    pretrain_root = task_root / "pretrain"
    tokenizer_path = pretrain_root / "tokenizer.json"
    source_train = _load_table(
        data_root / "pretrain-split" / "train_physical_pairs.csv"
    )
    source_validation = _load_table(
        data_root / "pretrain-split" / "validation_physical_pairs.csv"
    )
    source_train = _development_subset(
        source_train,
        task,
        args.pretrain_limit_per_class,
        seed,
    )
    source_validation = _development_subset(
        source_validation,
        task,
        max(1, args.pretrain_limit_per_class // 4)
        if args.pretrain_limit_per_class
        else 0,
        seed + 1000,
    )
    if args.resume and (pretrain_root / "best_model.pt").is_file():
        tokenizer = SmilesTokenizer.load(tokenizer_path)
        print(f"[resume] {task} seed-{seed} pretraining")
    else:
        pretrain_root.mkdir(parents=True, exist_ok=True)
        tokenizer = SmilesTokenizer.from_smiles(
            list(source_train["reactant_A"])
            + list(source_train["reactant_B"]),
            max_length=args.max_length,
        )
        tokenizer.save(tokenizer_path)
        model = _model(tokenizer, task).to(device)
        _fit(
            model,
            source_train,
            source_validation,
            tokenizer,
            task=task,
            epochs=args.pretrain_epochs,
            patience=args.pretrain_patience,
            batch_size=args.batch_size,
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
            beta=args.effective_number_beta,
            variants=args.smiles_variants,
            seed=seed,
            device=device,
            output_dir=pretrain_root,
            tensorboard_dir=(
                str(Path(args.tensorboard_root) / task / f"seed-{seed}" / "pretrain")
                if args.tensorboard_root
                else None
            ),
        )

    target_table_path = data_root / "four_class_target_physical_pairs.csv"
    target = _load_table(target_table_path)
    zero_prediction = pretrain_root / "target_zero_shot_predictions.csv"
    if not (args.resume and zero_prediction.is_file()):
        model = _model(tokenizer, task).to(device)
        model.load_state_dict(_load_state(pretrain_root / "best_model.pt", device))
        evaluation = _evaluate(
            model,
            target,
            tokenizer,
            task,
            args.batch_size,
            device,
        )
        _write_predictions(zero_prediction, target, evaluation, task)
    _run_prediction_summary(
        root,
        task,
        target_table_path,
        [zero_prediction],
        pretrain_root / "target-zero-shot-summary",
        seed,
    )

    fold_predictions = []
    for fold in range(args.folds):
        fold_seed = seed + fold
        fold_data = data_root / "folds" / f"fold-{fold}"
        target_train = _load_table(fold_data / "train_physical_pairs.csv")
        target_test = _load_table(fold_data / "test_physical_pairs.csv")
        split = grouped_stratified_split(
            task_labels(target_train, task),
            target_train["pair_key"].to_numpy(),
            validation_fraction=args.validation_fraction,
            seed=fold_seed,
        )
        selection_train = target_train.iloc[list(split.train_indices)].reset_index(
            drop=True
        )
        selection_validation = target_train.iloc[
            list(split.validation_indices)
        ].reset_index(drop=True)
        fold_root = task_root / f"fold-{fold}"
        selection_root = fold_root / "selection"
        final_root = fold_root / "final"
        selection_path = selection_root / "selection_result.json"
        if args.resume and selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            print(f"[resume] {task} seed-{seed} fold-{fold} selection")
        else:
            selection_model = _model(tokenizer, task).to(device)
            selection_model.load_state_dict(
                _load_state(pretrain_root / "best_model.pt", device)
            )
            selection_model.freeze_encoders()
            selection = _fit(
                selection_model,
                selection_train,
                selection_validation,
                tokenizer,
                task=task,
                epochs=args.finetune_epochs,
                patience=args.finetune_patience,
                batch_size=args.batch_size,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
                beta=args.effective_number_beta,
                variants=args.smiles_variants,
                seed=fold_seed,
                device=device,
                output_dir=selection_root,
                tensorboard_dir=(
                    str(
                        Path(args.tensorboard_root)
                        / task
                        / f"seed-{seed}"
                        / f"fold-{fold}"
                        / "selection"
                    )
                    if args.tensorboard_root
                    else None
                ),
            )
            split.manifest.to_csv(
                selection_root / "split_manifest.csv",
                index=False,
            )
        final_checkpoint = final_root / "best_model.pt"
        if args.resume and final_checkpoint.is_file():
            print(f"[resume] {task} seed-{seed} fold-{fold} final")
        else:
            final_model = _model(tokenizer, task).to(device)
            final_model.load_state_dict(
                _load_state(pretrain_root / "best_model.pt", device)
            )
            final_model.freeze_encoders()
            _fit(
                final_model,
                target_train,
                None,
                tokenizer,
                task=task,
                epochs=int(selection["best_epoch"]),
                patience=0,
                batch_size=args.batch_size,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
                beta=args.effective_number_beta,
                variants=args.smiles_variants,
                seed=fold_seed,
                device=device,
                output_dir=final_root,
                tensorboard_dir=(
                    str(
                        Path(args.tensorboard_root)
                        / task
                        / f"seed-{seed}"
                        / f"fold-{fold}"
                        / "final"
                    )
                    if args.tensorboard_root
                    else None
                ),
            )
        prediction = final_root / "target_test_predictions.csv"
        if not (args.resume and prediction.is_file()):
            final_model = _model(tokenizer, task).to(device)
            final_model.load_state_dict(_load_state(final_checkpoint, device))
            evaluation = _evaluate(
                final_model,
                target_test,
                tokenizer,
                task,
                args.batch_size,
                device,
            )
            _write_predictions(prediction, target_test, evaluation, task)
        fold_predictions.append(prediction)

    _run_prediction_summary(
        root,
        task,
        target_table_path,
        fold_predictions,
        task_root / "oof-summary",
        seed,
        allow_incomplete=args.folds < 5,
    )


def main():
    args = parse_args()
    if args.folds < 1:
        raise ValueError("--folds must be positive")
    root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _parse_list(args.tasks)
    seeds = _parse_list(args.seeds, int)
    unknown_tasks = set(tasks).difference(CLASS_NAMES)
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {sorted(unknown_tasks)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "schema_version": "mcc-gcn-target-api-smiles-cnn-run-v1",
        "model": "independent_dual_branch_smiles_cnn",
        "representation": "randomized_smiles",
        "data_root": str(data_root),
        "tasks": tasks,
        "seeds": seeds,
        "folds": args.folds,
        "max_length": args.max_length,
        "smiles_variants": args.smiles_variants,
        "batch_size": args.batch_size,
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "pretrain_patience": args.pretrain_patience,
        "finetune_patience": args.finetune_patience,
        "pretrain_lr": args.pretrain_lr,
        "finetune_lr": args.finetune_lr,
        "weight_decay": args.weight_decay,
        "effective_number_beta": args.effective_number_beta,
        "validation_fraction": args.validation_fraction,
        "class_weighting": "effective-number",
        "fine_tuning": "frozen_smiles_encoders_train_classifier",
        "orientation_training": "both_orders_after_physical_pair_split",
        "orientation_inference": "mean_probability",
        "pretrain_limit_per_class": args.pretrain_limit_per_class,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (output_root / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2, sort_keys=True))
    for seed in seeds:
        seed_everything(seed)
        for task in tasks:
            _run_task_seed(
                args,
                root,
                data_root,
                output_root,
                task,
                seed,
                device,
            )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "summarize_target_api_runs.py"),
            "--run-dir",
            str(output_root),
            "--output-dir",
            str(output_root / "multiseed-summary"),
        ],
        check=True,
    )
    print(f"SMILES CNN target-API experiment complete: {output_root}")


if __name__ == "__main__":
    main()

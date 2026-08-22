#!/usr/bin/env python3
"""Run the official DeepCocrystal architecture on the frozen binary task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/revision-three-api",
    )
    parser.add_argument(
        "--deepcocrystal-repo",
        required=True,
        help="Unmodified checkout of https://github.com/molML/deep-cocrystal.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smiles-variants", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tensorboard-dir")
    return parser.parse_args()


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, keep_default_na=False)
    required = {"reactant_A", "reactant_B", "label_int", "pair_key"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if table["pair_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate physical pairs")
    return table


def _binary_labels(table: pd.DataFrame) -> np.ndarray:
    return (table["label_int"].to_numpy(dtype=np.int64) != 0).astype(
        np.int64
    )


def _clean_smiles(values, clean_smiles):
    cleaned = [
        clean_smiles(
            str(value),
            uncharge=True,
            remove_stereochemistry=True,
            to_canonical=True,
        )
        for value in values
    ]
    if any(value is None for value in cleaned):
        raise ValueError("DeepCocrystal preprocessing rejected a SMILES")
    return cleaned


def _randomized_variants(smiles: str, count: int, seed: int) -> list[str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid cleaned SMILES: {smiles}")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    if count == 1:
        return [canonical]
    rng = np.random.default_rng(seed)
    roots = rng.permutation(molecule.GetNumAtoms())
    generated = [canonical]
    generated.extend(
        Chem.MolToSmiles(
            molecule,
            canonical=False,
            rootedAtAtom=int(root),
            isomericSmiles=False,
        )
        for root in roots
    )
    unique = list(dict.fromkeys(generated))
    return [unique[index % len(unique)] for index in range(count)]


def _training_arrays(table, labels, variants, seed, clean_smiles):
    left = _clean_smiles(table["reactant_A"], clean_smiles)
    right = _clean_smiles(table["reactant_B"], clean_smiles)
    output_left = []
    output_right = []
    output_labels = []
    for index, (first, second, label) in enumerate(
        zip(left, right, labels)
    ):
        first_variants = _randomized_variants(
            first,
            variants,
            seed + index * 2,
        )
        second_variants = _randomized_variants(
            second,
            variants,
            seed + index * 2 + 1,
        )
        for reverse in (False, True):
            for variant in range(variants):
                a = first_variants[variant]
                b = second_variants[(variant + 1) % variants]
                output_left.append(b if reverse else a)
                output_right.append(a if reverse else b)
                output_labels.append(int(label))
    return output_left, output_right, np.asarray(output_labels, dtype=np.int64)


def _evaluation_arrays(table, clean_smiles):
    left = np.asarray(_clean_smiles(table["reactant_A"], clean_smiles))
    right = np.asarray(_clean_smiles(table["reactant_B"], clean_smiles))
    return left, right


def _space_separate(values, converter):
    return converter(list(values))


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both binary classes are required")
    total = int(counts.sum())
    return {
        label: total / (2.0 * int(count))
        for label, count in enumerate(counts)
    }


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    args = parse_args()
    if args.smiles_variants < 1:
        raise ValueError("--smiles-variants must be positive")
    data_root = Path(args.data_root).resolve()
    official_repo = Path(args.deepcocrystal_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (official_repo / "deepcocrystal" / "deepcocrystal.py").is_file():
        raise FileNotFoundError(
            f"Not an official DeepCocrystal checkout: {official_repo}"
        )
    sys.path.insert(0, str(official_repo))

    try:
        import tensorflow as tf
        from deepcocrystal import smiles_preprocessing
        from deepcocrystal.deepcocrystal import DeepCocrystal
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow and an unmodified DeepCocrystal checkout are required"
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)
    train_path = data_root / "pretrain-split" / "train_physical_pairs.csv"
    validation_path = (
        data_root / "pretrain-split" / "validation_physical_pairs.csv"
    )
    target_path = data_root / "four_class_target_physical_pairs.csv"
    train = _load_table(train_path)
    validation = _load_table(validation_path)
    target = _load_table(target_path)
    train_labels = _binary_labels(train)
    validation_labels = _binary_labels(validation)
    target_labels = _binary_labels(target)

    train_left, train_right, augmented_train_labels = _training_arrays(
        train,
        train_labels,
        args.smiles_variants,
        args.seed,
        smiles_preprocessing.clean_smiles,
    )
    validation_left, validation_right, augmented_validation_labels = (
        _training_arrays(
            validation,
            validation_labels,
            1,
            args.seed + 100000,
            smiles_preprocessing.clean_smiles,
        )
    )
    train_left = _space_separate(
        train_left,
        smiles_preprocessing.space_separate_smiles_list,
    )
    train_right = _space_separate(
        train_right,
        smiles_preprocessing.space_separate_smiles_list,
    )
    validation_left = _space_separate(
        validation_left,
        smiles_preprocessing.space_separate_smiles_list,
    )
    validation_right = _space_separate(
        validation_right,
        smiles_preprocessing.space_separate_smiles_list,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with _working_directory(official_repo):
        model = DeepCocrystal()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / "training_history.csv")
        ),
    ]
    if args.tensorboard_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(args.tensorboard_dir))
    history = model.fit(
        x=[train_left, train_right],
        y=augmented_train_labels,
        validation_data=(
            [validation_left, validation_right],
            augmented_validation_labels,
        ),
        class_weight=_class_weights(augmented_train_labels),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )
    model.save_weights(output_dir / "best_model.weights.h5")

    target_left, target_right = _evaluation_arrays(
        target,
        smiles_preprocessing.clean_smiles,
    )
    target_left_tokens = _space_separate(
        target_left,
        smiles_preprocessing.space_separate_smiles_list,
    )
    target_right_tokens = _space_separate(
        target_right,
        smiles_preprocessing.space_separate_smiles_list,
    )
    forward = model.predict(
        [target_left_tokens, target_right_tokens],
        batch_size=args.batch_size,
        verbose=1,
    ).reshape(-1)
    reverse = model.predict(
        [target_right_tokens, target_left_tokens],
        batch_size=args.batch_size,
        verbose=1,
    ).reshape(-1)
    probabilities = (forward + reverse) / 2.0
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = float(np.mean(predictions == target_labels))
    balanced_accuracy = float(
        balanced_accuracy_score(target_labels, predictions)
    )

    prediction_table = pd.DataFrame(
        {
            "Pair Key": target["pair_key"],
            "True Label": target_labels,
            "Predicted Label": predictions,
            "P(positive)": probabilities,
            "P(A_B,positive)": forward,
            "P(B_A,positive)": reverse,
        }
    )
    prediction_table.to_csv(
        output_dir / "target_predictions.csv",
        index=False,
    )
    summary = {
        "schema_version": "mcc-gcn-deepcocrystal-architecture-baseline-v1",
        "task": "binary_any_mcc_vs_negative",
        "official_repository": "https://github.com/molML/deep-cocrystal",
        "official_commit": _git_head(official_repo),
        "official_pretrained_checkpoint_used": False,
        "reason_checkpoint_not_used": (
            "The official training data contain the three target APIs and "
            "use a cocrystal-specific binary label."
        ),
        "source_train_physical_pairs": len(train),
        "source_validation_physical_pairs": len(validation),
        "target_physical_pairs": len(target),
        "smiles_variants": args.smiles_variants,
        "orientation_training": "both_orders_after_physical_pair_split",
        "orientation_inference": "mean_probability",
        "class_weighting": "inverse_frequency",
        "epochs_completed": len(history.history["loss"]),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": confusion_matrix(
            target_labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "tensorflow_version": tf.__version__,
        "rdkit_version": rdBase.rdkitVersion,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

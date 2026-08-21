#!/usr/bin/env python3
"""Run RDKit descriptor baselines for the three-API revision experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcc_gcn.data.descriptors import (
    MOLECULE_DESCRIPTOR_NAMES,
    featurize_pairs,
)
from scripts.summarize_target_api_predictions import (
    CLASS_NAMES,
    _metrics,
    sha256_file,
)


MODEL_NAMES = ("svm", "random_forest", "mlp")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/revision-three-api",
        help="Root containing frozen pretraining splits and target folds.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="binary,four-class")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--source-limit-per-class", type=int)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--rf-estimators", type=int, default=500)
    parser.add_argument("--mlp-max-iter", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _labels(table: pd.DataFrame, task: str) -> np.ndarray:
    labels = table["label_int"].to_numpy(dtype=np.int64)
    if task == "binary":
        return np.where(labels == 0, 0, 1)
    return labels


def _features(
    table: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    return featurize_pairs(
        table[["reactant_A", "reactant_B"]].itertuples(
            index=False,
            name=None,
        ),
        cache=cache,
    )


def _limit_per_class(
    table: pd.DataFrame,
    task: str,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    if limit is None:
        return table.reset_index(drop=True)
    labels = _labels(table, task)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) > limit:
            indices = rng.choice(indices, size=limit, replace=False)
        selected.extend(indices.tolist())
    return table.iloc[sorted(selected)].reset_index(drop=True)


def _balanced_resample(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    target = int(counts.max())
    indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == label),
                size=target,
                replace=count < target,
            )
            for label, count in zip(classes, counts)
        ]
    )
    rng.shuffle(indices)
    return features[indices], labels[indices]


def _fit_classifier(
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    args,
):
    scaler = None
    train_features = features
    weights = compute_sample_weight(class_weight="balanced", y=labels)
    if model_name in {"svm", "mlp"}:
        scaler = StandardScaler().fit(features)
        train_features = scaler.transform(features)

    if model_name == "svm":
        classifier = SVC(
            C=args.svm_c,
            gamma="scale",
            kernel="rbf",
            cache_size=4096,
            random_state=seed,
        )
        classifier.fit(train_features, labels, sample_weight=weights)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
        classifier.fit(train_features, labels, sample_weight=weights)
    elif model_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )
        if "sample_weight" in inspect.signature(classifier.fit).parameters:
            classifier.fit(
                train_features,
                labels,
                sample_weight=weights,
            )
        else:
            balanced_x, balanced_y = _balanced_resample(
                train_features,
                labels,
                seed,
            )
            classifier.fit(balanced_x, balanced_y)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return scaler, classifier


def _predict(scaler, classifier, features, num_classes):
    model_features = scaler.transform(features) if scaler else features
    predictions = classifier.predict(model_features).astype(np.int64)
    probabilities = None
    supports_probability = not isinstance(classifier, SVC) or bool(
        classifier.probability
    )
    if supports_probability and hasattr(classifier, "predict_proba"):
        probabilities = np.zeros(
            (len(features), num_classes),
            dtype=np.float64,
        )
        raw = classifier.predict_proba(model_features)
        for index, label in enumerate(classifier.classes_):
            probabilities[:, int(label)] = raw[:, index]
    return predictions, probabilities


def _prediction_table(
    table: pd.DataFrame,
    task: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    test_fold: int | None,
) -> pd.DataFrame:
    class_names = CLASS_NAMES[task]
    pair_key_column = (
        "representative_pair_key"
        if "representative_pair_key" in table
        else "pair_key"
    )
    columns = {
        "Pair Key": table[pair_key_column].to_numpy(),
        "True Label": _labels(table, task),
        "Predicted Label": predictions,
        "Correct": predictions == _labels(table, task),
    }
    if probabilities is not None:
        columns.update(
            {
                f"P({name})": probabilities[:, index]
                for index, name in enumerate(class_names)
            }
        )
    result = pd.DataFrame(columns)
    if test_fold is not None:
        result["test_fold"] = test_fold
    return result


def _write_target_summary(
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    task: str,
    output_dir: Path,
    *,
    bootstrap_replicates: int,
    seed: int,
    target_path: Path,
    prediction_paths: list[Path],
):
    expected = target.rename(
        columns={"representative_pair_key": "Pair Key"}
    )
    combined = predictions.merge(
        expected[["Pair Key", "target_apis", "label_int"]],
        on="Pair Key",
        how="left",
        validate="one_to_one",
    )
    if (
        combined["target_apis"].isna().any()
        or combined["target_apis"].eq("").any()
    ):
        raise ValueError("Predictions contain unknown target pairs")
    if not np.array_equal(
        combined["True Label"].to_numpy(dtype=np.int64),
        _labels(combined, task),
    ):
        raise ValueError("Prediction labels disagree with target metadata")

    class_names = CLASS_NAMES[task]
    summary = {
        "schema_version": "mcc-gcn-descriptor-baseline-summary-v1",
        "task": task,
        "class_names": class_names,
        "pooled": _metrics(
            combined,
            class_names,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        ),
        "by_target_api": {},
        "inputs": {
            "target_table": {
                "path": str(target_path),
                "sha256": sha256_file(target_path),
            },
            "prediction_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in prediction_paths
            ],
        },
    }
    api_names = sorted(
        {
            name
            for values in target["target_apis"].str.split(";")
            for name in values
            if name
        }
    )
    for index, api_name in enumerate(api_names):
        subset = combined.loc[
            combined["target_apis"].str.split(";").map(
                lambda names: api_name in names
            )
        ]
        summary["by_target_api"][api_name] = _metrics(
            subset,
            class_names,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index + 1,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "combined_predictions.csv", index=False)
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _summary_row(model_name, task, stage, seed, summary):
    pooled = summary["pooled"]
    row = {
        "model": model_name,
        "task": task,
        "stage": stage,
        "seed": seed,
        "pairs": pooled["pairs"],
        "accuracy": pooled["accuracy"],
        "balanced_accuracy": pooled["balanced_accuracy_present_classes"],
    }
    for api_name, values in summary["by_target_api"].items():
        key = api_name.lower().replace(" ", "_")
        row[f"api_{key}_accuracy"] = values["accuracy"]
        row[f"api_{key}_balanced_accuracy"] = values[
            "balanced_accuracy_present_classes"
        ]
    return row


def _write_aggregate(output_dir: Path, rows: list[dict]):
    raw = pd.DataFrame(rows).sort_values(["model", "task", "stage", "seed"])
    raw.to_csv(output_dir / "baseline_runs.csv", index=False)
    aggregate_rows = []
    metadata = {"model", "task", "stage", "seed"}
    for keys, group in raw.groupby(["model", "task", "stage"], sort=True):
        result = {
            "model": keys[0],
            "task": keys[1],
            "stage": keys[2],
            "runs": len(group),
            "seeds": ",".join(str(value) for value in group["seed"]),
        }
        for column in sorted(set(raw.columns).difference(metadata)):
            values = group[column].dropna().astype(float).tolist()
            if not values:
                continue
            result[f"{column}_mean"] = statistics.fmean(values)
            result[f"{column}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(result)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "baseline_aggregate.csv", index=False)
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "mcc-gcn-descriptor-baselines-v1",
                "runs": rows,
                "aggregate": aggregate_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Three-API RDKit descriptor baselines",
        "",
        "| Model | Task | Training regime | Seeds | Accuracy | Balanced accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["task"],
                    row["stage"],
                    str(row["runs"]),
                    f"{row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}",
                    (
                        f"{row['balanced_accuracy_mean']:.3f} +/- "
                        f"{row['balanced_accuracy_std']:.3f}"
                    ),
                ]
            )
            + " |"
        )
    (output_dir / "baseline_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    tasks = _comma_values(args.tasks)
    models = _comma_values(args.models)
    seeds = [int(value) for value in _comma_values(args.seeds)]
    unknown_tasks = set(tasks).difference(CLASS_NAMES)
    unknown_models = set(models).difference(MODEL_NAMES)
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    source_path = data_root / "pretrain-split" / "train_physical_pairs.csv"
    target_path = data_root / "four_class_target_physical_pairs.csv"
    source = pd.read_csv(source_path, keep_default_na=False)
    target = pd.read_csv(target_path, keep_default_na=False)
    folds = [
        (
            pd.read_csv(
                data_root / "folds" / f"fold-{fold}" / "train_physical_pairs.csv",
                keep_default_na=False,
            ),
            pd.read_csv(
                data_root / "folds" / f"fold-{fold}" / "test_physical_pairs.csv",
                keep_default_na=False,
            ),
        )
        for fold in range(args.folds)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "mcc-gcn-descriptor-baseline-run-v1",
        "data_root": str(data_root),
        "source_table": str(source_path),
        "target_table": str(target_path),
        "models": models,
        "tasks": tasks,
        "seeds": seeds,
        "folds": args.folds,
        "feature_representation": "rdkit_2d_mean_and_absolute_difference",
        "molecule_descriptors": list(MOLECULE_DESCRIPTOR_NAMES),
        "source_limit_per_class": args.source_limit_per_class,
        "svm_c": args.svm_c,
        "rf_estimators": args.rf_estimators,
        "mlp_max_iter": args.mlp_max_iter,
        "scikit_learn_version": sklearn.__version__,
        "rdkit_version": rdBase.rdkitVersion,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    descriptor_cache: dict[str, np.ndarray] = {}
    target_features = _features(target, descriptor_cache)
    fold_features = [
        (
            _features(train, descriptor_cache),
            _features(test, descriptor_cache),
        )
        for train, test in folds
    ]
    result_rows = []
    for model_name in models:
        for task in tasks:
            num_classes = len(CLASS_NAMES[task])
            for seed in seeds:
                print(f"\n[{model_name}] task={task} seed={seed}")
                seed_dir = output_dir / model_name / task / f"seed-{seed}"
                limited_source = _limit_per_class(
                    source,
                    task,
                    args.source_limit_per_class,
                    seed,
                )
                source_features = _features(limited_source, descriptor_cache)
                source_labels = _labels(limited_source, task)
                zero_path = seed_dir / "pretrain" / "target_zero_shot_predictions.csv"
                zero_summary_path = (
                    seed_dir
                    / "pretrain"
                    / "target-zero-shot-summary"
                    / "summary_metrics.json"
                )
                if args.resume and zero_path.is_file() and zero_summary_path.is_file():
                    zero_predictions = pd.read_csv(zero_path, keep_default_na=False)
                    zero_summary = json.loads(
                        zero_summary_path.read_text(encoding="utf-8")
                    )
                else:
                    scaler, classifier = _fit_classifier(
                        model_name,
                        source_features,
                        source_labels,
                        seed,
                        args,
                    )
                    predicted, probabilities = _predict(
                        scaler,
                        classifier,
                        target_features,
                        num_classes,
                    )
                    zero_predictions = _prediction_table(
                        target,
                        task,
                        predicted,
                        probabilities,
                        test_fold=None,
                    )
                    zero_path.parent.mkdir(parents=True, exist_ok=True)
                    zero_predictions.to_csv(zero_path, index=False)
                    zero_summary = _write_target_summary(
                        zero_predictions,
                        target,
                        task,
                        zero_summary_path.parent,
                        bootstrap_replicates=args.bootstrap_replicates,
                        seed=seed,
                        target_path=target_path,
                        prediction_paths=[zero_path],
                    )
                result_rows.append(
                    _summary_row(
                        model_name,
                        task,
                        "source_zero_shot",
                        seed,
                        zero_summary,
                    )
                )

                fold_predictions = []
                fold_paths = []
                for fold, ((train, test), (train_x, test_x)) in enumerate(
                    zip(folds, fold_features)
                ):
                    prediction_path = (
                        seed_dir
                        / f"fold-{fold}"
                        / "target-supervised"
                        / "target_test_predictions.csv"
                    )
                    fold_paths.append(prediction_path)
                    if args.resume and prediction_path.is_file():
                        fold_prediction = pd.read_csv(
                            prediction_path,
                            keep_default_na=False,
                        )
                    else:
                        scaler, classifier = _fit_classifier(
                            model_name,
                            train_x,
                            _labels(train, task),
                            seed + fold,
                            args,
                        )
                        predicted, probabilities = _predict(
                            scaler,
                            classifier,
                            test_x,
                            num_classes,
                        )
                        fold_prediction = _prediction_table(
                            test,
                            task,
                            predicted,
                            probabilities,
                            test_fold=fold,
                        )
                        prediction_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        fold_prediction.to_csv(prediction_path, index=False)
                    fold_predictions.append(fold_prediction)
                oof = pd.concat(fold_predictions, ignore_index=True)
                oof_summary_dir = seed_dir / "target-supervised-oof-summary"
                oof_summary_path = oof_summary_dir / "summary_metrics.json"
                if args.resume and oof_summary_path.is_file():
                    oof_summary = json.loads(
                        oof_summary_path.read_text(encoding="utf-8")
                    )
                else:
                    oof_summary = _write_target_summary(
                        oof,
                        target,
                        task,
                        oof_summary_dir,
                        bootstrap_replicates=args.bootstrap_replicates,
                        seed=seed,
                        target_path=target_path,
                        prediction_paths=fold_paths,
                    )
                result_rows.append(
                    _summary_row(
                        model_name,
                        task,
                        "target_supervised_5fold",
                        seed,
                        oof_summary,
                    )
                )
                print(
                    "  zero-shot BACC="
                    f"{zero_summary['pooled']['balanced_accuracy_present_classes']:.4f}; "
                    "target-supervised BACC="
                    f"{oof_summary['pooled']['balanced_accuracy_present_classes']:.4f}"
                )

    _write_aggregate(output_dir, result_rows)
    print(f"\nDescriptor baseline experiment complete: {output_dir}")


if __name__ == "__main__":
    main()

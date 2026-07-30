"""Evaluate MCC-GCN by averaging matched A/B and B/A predictions."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcc_gcn.data.dataset import GraphDataLoader
from mcc_gcn.models.gcn import GCNNet
from mcc_gcn.models.metrics import calculate_detailed_metrics, calculate_metrics
from mcc_gcn.utils import resolve_npz, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data-1", required=True, help="A/B order NPZ")
    parser.add_argument("--test-data-2", required=True, help="B/A order NPZ")
    parser.add_argument("--mol-blocks", default="data/HKU_data.pkl.gz")
    parser.add_argument("--rebuild-features", action="store_true")
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
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="prediction_results.csv")
    return parser.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_test_data(name, mol_blocks, rebuild, task):
    npz_path = resolve_npz(name, mol_blocks, rebuild=rebuild)
    label_mode = "binary" if task == "binary" else "stored"
    items = GraphDataLoader(
        npz_file=npz_path,
        label_mode=label_mode,
    ).pyg_data
    return npz_path, items


def _json_compatible_metrics(metrics):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }


def _batch_pair_keys(batch):
    value = getattr(batch, "pair_key", None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def main():
    args = parse_args()
    expected_num_classes = 2 if args.task == "binary" else 4
    if (
        args.num_classes is not None
        and args.num_classes != expected_num_classes
    ):
        raise ValueError("--num-classes does not agree with --task")
    num_classes = expected_num_classes

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_npz_1, test_items_1 = _load_test_data(
        args.test_data_1,
        args.mol_blocks,
        args.rebuild_features,
        args.task,
    )
    test_npz_2, test_items_2 = _load_test_data(
        args.test_data_2,
        args.mol_blocks,
        args.rebuild_features,
        args.task,
    )
    if len(test_items_1) != len(test_items_2):
        raise ValueError(
            "A/B and B/A test files contain different row counts: "
            f"{len(test_items_1)} != {len(test_items_2)}"
        )

    model = GCNNet(
        num_classes=num_classes,
        model_size=args.model_size,
    ).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()

    loader_1 = DataLoader(test_items_1, batch_size=1, shuffle=False)
    loader_2 = DataLoader(test_items_2, batch_size=1, shuffle=False)
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_pair_keys = []
    pair_keys_present = None
    with torch.no_grad():
        for row_index, (batch_1, batch_2) in enumerate(
            zip(loader_1, loader_2)
        ):
            labels_1 = batch_1.y.cpu().numpy()
            labels_2 = batch_2.y.cpu().numpy()
            if not np.array_equal(labels_1, labels_2):
                raise ValueError(
                    f"A/B and B/A labels differ at row {row_index}"
                )
            pair_key_1 = _batch_pair_keys(batch_1)
            pair_key_2 = _batch_pair_keys(batch_2)
            if (pair_key_1 is None) != (pair_key_2 is None):
                raise ValueError(
                    "Only one test orientation contains pair_keys"
                )
            row_has_pair_keys = pair_key_1 is not None
            if pair_keys_present is None:
                pair_keys_present = row_has_pair_keys
            elif pair_keys_present != row_has_pair_keys:
                raise ValueError(
                    "Pair-key presence changes within the test dataset"
                )
            if pair_key_1 is not None:
                if pair_key_1 != pair_key_2:
                    raise ValueError(
                        f"A/B and B/A pair_keys differ at row {row_index}"
                    )
                all_pair_keys.extend(pair_key_1)

            batch_1 = batch_1.to(device)
            batch_2 = batch_2.to(device)
            probabilities_1 = F.softmax(
                model(batch_1.x, batch_1.edge_index, batch_1.batch),
                dim=1,
            )
            probabilities_2 = F.softmax(
                model(batch_2.x, batch_2.edge_index, batch_2.batch),
                dim=1,
            )
            probabilities = np.average(
                [
                    probabilities_1.cpu().numpy(),
                    probabilities_2.cpu().numpy(),
                ],
                axis=0,
            )
            all_predictions.extend(np.argmax(probabilities, axis=1))
            all_labels.extend(labels_1)
            all_probabilities.extend(probabilities)

    predictions = np.asarray(all_predictions)
    labels = np.asarray(all_labels)
    probabilities = np.asarray(all_probabilities)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            f"Test labels fall outside configured {num_classes}-class task"
        )
    class_names = (
        ["negative", "positive"]
        if args.task == "binary"
        else ["negative", "salt", "cocrystal", "hydrate_or_solvate"]
    )
    per_class_accuracy, balanced_accuracy, overall_accuracy = calculate_metrics(
        labels,
        predictions,
        num_classes,
    )
    detailed = calculate_detailed_metrics(
        labels,
        predictions,
        num_classes,
    )
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes)),
    )

    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"\nConfusion Matrix:\n{matrix}")
    for index, name in enumerate(class_names):
        print(f"{name} Accuracy: {per_class_accuracy[index]:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_columns = {
        "True Label": labels,
        "Predicted Label": predictions,
        "Correct": predictions == labels,
        **{
            f"P({class_names[index]})": probabilities[:, index]
            for index in range(num_classes)
        },
    }
    if all_pair_keys:
        result_columns = {"Pair Key": all_pair_keys, **result_columns}
    pd.DataFrame(result_columns).to_csv(output_path, index=False)

    metrics = {
        "schema_version": "mcc-gcn-evaluation-v1",
        "task": args.task,
        "num_classes": num_classes,
        "rows": len(labels),
        "overall_accuracy": float(overall_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "per_class_accuracy": {
            name: float(per_class_accuracy[index])
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "detailed_metrics": _json_compatible_metrics(detailed),
        "model": {
            "path": args.model,
            "sha256": _sha256_file(args.model),
        },
        "test_data": [
            {"path": test_npz_1, "sha256": _sha256_file(test_npz_1)},
            {"path": test_npz_2, "sha256": _sha256_file(test_npz_2)},
        ],
        "predictions": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
        },
        "device": str(device),
        "torch_version": torch.__version__,
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

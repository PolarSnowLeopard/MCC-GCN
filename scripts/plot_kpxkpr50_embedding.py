#!/usr/bin/env python3
"""Plot post-fine-tuning KPXKPR-50 latent representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcc_gcn.data.dataset import GraphDataLoader
from mcc_gcn.data.legacy import load_legacy_dense_items
from mcc_gcn.models.gcn import GCNNet
from mcc_gcn.utils import seed_everything

CLASS_NAMES = ["Negative", "Salt", "Cocrystal", "Hydrate/solvate"]
CLASS_COLORS = ["#6B7280", "#D97706", "#2E8B57", "#277DA1"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data-1", required=True, help="A/B order NPZ")
    parser.add_argument("--test-data-2", required=True, help="B/A order NPZ")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-mode",
        choices=["legacy-padded", "trimmed"],
        default="legacy-padded",
        help="Use legacy padding for the submitted KPXKPR-50 checkpoint.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "pca", "tsne"],
        default="umap",
    )
    parser.add_argument(
        "--representation",
        choices=["graph", "fc1", "fc2", "logits", "probabilities"],
        default="fc2",
        help="Representation used for the two-dimensional projection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)
    return parser.parse_args()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_padded_items(path: str | Path):
    return load_legacy_dense_items(path, preserve_padding=True)


def _load_items(path: str | Path, input_mode: str) -> list[Data]:
    if input_mode == "legacy-padded":
        return _legacy_padded_items(path)
    return GraphDataLoader(npz_file=path).pyg_data


def _infer(model, items, device, batch_size):
    representations = {"graph": [], "fc1": [], "fc2": []}

    def capture(name):
        def hook(_module, inputs):
            representations[name].append(inputs[0].detach().cpu())

        return hook

    handles = [
        model.fc1.register_forward_pre_hook(capture("graph")),
        model.fc2.register_forward_pre_hook(capture("fc1")),
        model.fc_out.register_forward_pre_hook(capture("fc2")),
    ]
    logits_batches = []
    probabilities = []
    labels = []
    try:
        with torch.no_grad():
            for batch in DataLoader(
                items,
                batch_size=batch_size,
                shuffle=False,
            ):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                logits_batches.append(logits.cpu())
                probabilities.append(logits.softmax(dim=1).cpu())
                labels.append(batch.y.cpu())
    finally:
        for handle in handles:
            handle.remove()
    probabilities = torch.cat(probabilities)
    representations = {
        name: torch.cat(batches).numpy()
        for name, batches in representations.items()
    }
    representations["logits"] = torch.cat(logits_batches).numpy()
    representations["probabilities"] = probabilities.numpy()
    return (
        representations,
        probabilities.numpy(),
        torch.cat(labels).numpy(),
    )


def _cluster_metrics(features: np.ndarray, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        raise ValueError("Clustering metrics require at least two classes")
    return {
        "silhouette": float(silhouette_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
    }


def _reduce(features: np.ndarray, args) -> np.ndarray:
    standardized = StandardScaler().fit_transform(features)
    if args.method == "pca":
        return PCA(n_components=2).fit_transform(standardized)
    if args.method == "tsne":
        perplexity = min(15.0, max(2.0, (len(features) - 1) / 3))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        ).fit_transform(standardized)
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP output requires umap-learn; reinstall the project dependencies"
        ) from exc
    return umap.UMAP(
        n_components=2,
        n_neighbors=min(args.umap_neighbors, len(features) - 1),
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
    ).fit_transform(standardized)


def _plot(table: pd.DataFrame, output_dir: Path, method: str):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    figure, axis = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
    for label, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        subset = table.loc[table["true_label"].eq(label)]
        correct = subset.loc[subset["correct"]]
        incorrect = subset.loc[~subset["correct"]]
        axis.scatter(
            correct["component_1"],
            correct["component_2"],
            s=38,
            c=color,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
            label=f"{name} (n={len(subset)})",
        )
        if not incorrect.empty:
            axis.scatter(
                incorrect["component_1"],
                incorrect["component_2"],
                s=54,
                c=color,
                marker="X",
                edgecolors="#111827",
                linewidths=0.6,
                alpha=0.95,
            )
    prefix = method.upper() if method != "tsne" else "t-SNE"
    axis.set_xlabel(f"{prefix} 1")
    axis.set_ylabel(f"{prefix} 2")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8,
        handletextpad=0.4,
    )
    axis.text(
        0.01,
        0.01,
        "X: misclassified",
        transform=axis.transAxes,
        fontsize=7.5,
        color="#374151",
    )
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(
            output_dir / f"kpxkpr50_post_finetune_embedding.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNNet(num_classes=4, model_size="large").to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()

    items_ab = _load_items(args.test_data_1, args.input_mode)
    items_ba = _load_items(args.test_data_2, args.input_mode)
    if len(items_ab) != len(items_ba):
        raise ValueError("A/B and B/A files contain different row counts")
    representations_ab, probabilities_ab, labels_ab = _infer(
        model,
        items_ab,
        device,
        args.batch_size,
    )
    representations_ba, probabilities_ba, labels_ba = _infer(
        model,
        items_ba,
        device,
        args.batch_size,
    )
    if not np.array_equal(labels_ab, labels_ba):
        raise ValueError("A/B and B/A files contain different labels")

    representations = {
        name: 0.5 * (representations_ab[name] + representations_ba[name])
        for name in representations_ab
    }
    latent = representations[args.representation]
    probabilities = 0.5 * (probabilities_ab + probabilities_ba)
    predictions = probabilities.argmax(axis=1)
    coordinates = _reduce(latent, args)
    table = pd.DataFrame(
        {
            "row_index": np.arange(len(labels_ab)),
            "true_label": labels_ab,
            "true_class": [CLASS_NAMES[value] for value in labels_ab],
            "predicted_label": predictions,
            "predicted_class": [CLASS_NAMES[value] for value in predictions],
            "correct": predictions == labels_ab,
            "component_1": coordinates[:, 0],
            "component_2": coordinates[:, 1],
            **{
                f"latent_{index}": latent[:, index]
                for index in range(latent.shape[1])
            },
        }
    )
    table.to_csv(output_dir / "kpxkpr50_post_finetune_embedding.csv", index=False)
    metrics = {
        "schema_version": "mcc-gcn-kpxkpr50-embedding-v1",
        "input_mode": args.input_mode,
        "reduction": {
            "method": args.method,
            "representation": args.representation,
            "seed": args.seed,
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
        },
        "samples": len(labels_ab),
        "class_counts": {
            CLASS_NAMES[label]: int(np.count_nonzero(labels_ab == label))
            for label in range(len(CLASS_NAMES))
        },
        "classification": {
            "accuracy": float(np.mean(predictions == labels_ab)),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels_ab, predictions)
            ),
        },
        "clustering": {
            "standardized_model_spaces": {
                name: _cluster_metrics(
                    StandardScaler().fit_transform(values),
                    labels_ab,
                )
                for name, values in representations.items()
            },
            "two_dimensional_projection": _cluster_metrics(
                coordinates,
                labels_ab,
            ),
        },
        "orientation": {
            "mean_latent_l2_distance": float(
                np.linalg.norm(
                    representations_ab[args.representation]
                    - representations_ba[args.representation],
                    axis=1,
                ).mean()
            ),
            "prediction_agreement": float(
                np.mean(
                    probabilities_ab.argmax(axis=1)
                    == probabilities_ba.argmax(axis=1)
                )
            ),
        },
        "model": {"path": args.model, "sha256": _sha256(args.model)},
        "test_data": [
            {"path": args.test_data_1, "sha256": _sha256(args.test_data_1)},
            {"path": args.test_data_2, "sha256": _sha256(args.test_data_2)},
        ],
        "software": {"torch": torch.__version__},
    }
    (output_dir / "kpxkpr50_post_finetune_embedding.metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(table, output_dir, args.method)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"KPXKPR-50 embedding outputs ready: {output_dir}")


if __name__ == "__main__":
    main()

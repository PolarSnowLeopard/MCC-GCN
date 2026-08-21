"""Strict loaders for frozen dense feature artifacts from the submitted work."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def load_legacy_dense_items(
    path: str | Path,
    *,
    row_indices: Iterable[int] | None = None,
    pair_keys: Sequence[str] | None = None,
    preserve_padding: bool = True,
    label_mode: str = "stored",
) -> list[Data]:
    """Load selected rows from a frozen dense NPZ without changing features.

    ``preserve_padding`` reproduces the historical graph construction, where
    every padded row entered global mean pooling. Disabling it applies the
    corrected loader behavior and retains only ``graph_size`` observed nodes.
    """
    if label_mode not in {"stored", "binary"}:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    with np.load(path, allow_pickle=True) as data:
        required = {"V", "A", "labels"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Legacy feature file is missing {sorted(missing)}")
        if not preserve_padding and "graph_size" not in data:
            raise ValueError("Trimmed legacy loading requires graph_size")
        vertices = data["V"]
        adjacencies = data["A"]
        labels = data["labels"]
        graph_sizes = data.get("graph_size")

    if row_indices is None:
        indices = list(range(len(labels)))
    else:
        indices = [int(index) for index in row_indices]
    if pair_keys is not None and len(pair_keys) != len(indices):
        raise ValueError("pair_keys and row_indices differ in length")

    items = []
    for position, index in enumerate(indices):
        if index < 0 or index >= len(labels):
            raise IndexError(f"Legacy feature row is out of range: {index}")
        padded_width = int(vertices[index].shape[0])
        observed_nodes = (
            int(graph_sizes[index]) if graph_sizes is not None else padded_width
        )
        if observed_nodes < 1 or observed_nodes > padded_width:
            raise ValueError(
                f"Invalid graph_size={observed_nodes} at row {index} "
                f"for padded width {padded_width}"
            )
        node_count = padded_width if preserve_padding else observed_nodes
        vertex = vertices[index, :node_count]
        adjacency = adjacencies[index, :node_count, :, :node_count]
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        edge_index = torch.nonzero(adjacency.sum(dim=1), as_tuple=False).t()
        label = int(labels[index])
        if label_mode == "binary":
            label = 0 if label == 0 else 1
        item = Data(
            x=torch.as_tensor(vertex, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long),
        )
        item.source_row = index
        item.observed_graph_size = observed_nodes
        item.legacy_padded_width = padded_width
        if pair_keys is not None:
            item.pair_key = str(pair_keys[position])
        items.append(item)
    return items

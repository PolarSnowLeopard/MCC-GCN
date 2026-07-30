"""Feature-artifact lineage and subset recovery helpers."""

from __future__ import annotations

import hashlib

import numpy as np


def _storage_format(data):
    if "storage_format" not in data:
        return "dense_padded_v1"
    return str(np.asarray(data["storage_format"]).item())


def _sample_feature_arrays(data, index):
    storage_format = _storage_format(data)
    if storage_format == "dense_padded_v1":
        node_count = int(np.asarray(data["graph_size"][index]).item())
        vertex = np.asarray(data["V"][index])[:node_count]
        adjacency = np.asarray(data["A"][index])[
            :node_count,
            :,
            :node_count,
        ]
        return vertex, adjacency
    if storage_format != "packed_sparse_v1":
        raise ValueError(
            f"Unsupported graph storage format: {storage_format}"
        )

    node_start = int(data["node_ptr"][index])
    node_end = int(data["node_ptr"][index + 1])
    edge_start = int(data["edge_ptr"][index])
    edge_end = int(data["edge_ptr"][index + 1])
    vertex = np.asarray(data["V"][node_start:node_end])
    edge_index = np.asarray(
        data["edge_index"][:, edge_start:edge_end],
        dtype=np.int64,
    )
    edge_attr = np.asarray(data["edge_attr"][edge_start:edge_end])
    adjacency = np.zeros(
        (len(vertex), edge_attr.shape[1], len(vertex)),
        dtype=edge_attr.dtype,
    )
    adjacency[edge_index[0], :, edge_index[1]] = edge_attr
    return vertex, adjacency


def sample_feature_fingerprint(data, index):
    """Hash one unpadded graph sample independently of NPZ padded width."""
    vertex, adjacency = _sample_feature_arrays(data, index)

    digest = hashlib.sha256()
    for value in (
        vertex,
        adjacency,
        np.asarray(data["labels"][index]),
        np.asarray(data["subgraph_size"][index]),
    ):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_unique_fingerprint_index(data):
    """Index samples by fingerprint and reject ambiguous duplicates."""
    result = {}
    duplicate_fingerprints = set()
    for index in range(len(data["labels"])):
        fingerprint = sample_feature_fingerprint(data, index)
        if fingerprint in result:
            duplicate_fingerprints.add(fingerprint)
        result[fingerprint] = index
    if duplicate_fingerprints:
        raise ValueError(
            "Feature artifact contains "
            f"{len(duplicate_fingerprints)} duplicate sample fingerprints"
        )
    return result


def match_feature_subset(reference, subset):
    """Return each subset row's index in a reference feature artifact."""
    reference_index = build_unique_fingerprint_index(reference)
    matched = []
    for subset_index in range(len(subset["labels"])):
        fingerprint = sample_feature_fingerprint(subset, subset_index)
        if fingerprint not in reference_index:
            raise ValueError(
                f"Subset sample {subset_index} is absent from reference"
            )
        matched.append(reference_index[fingerprint])
    if len(set(matched)) != len(matched):
        raise ValueError("Subset maps multiple rows to one reference sample")
    return matched

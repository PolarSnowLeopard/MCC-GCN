"""Feature-artifact lineage and subset recovery helpers."""

from __future__ import annotations

import hashlib

import numpy as np


def sample_feature_fingerprint(data, index):
    """Hash one unpadded graph sample independently of NPZ padded width."""
    node_count = int(np.asarray(data["graph_size"][index]).item())
    vertex = np.asarray(data["V"][index])[:node_count]
    adjacency = np.asarray(data["A"][index])[
        :node_count,
        :,
        :node_count,
    ]

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

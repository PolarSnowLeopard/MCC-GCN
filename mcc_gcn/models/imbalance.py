"""Class-imbalance strategies that retain every training example."""

from __future__ import annotations

import numpy as np


def calculate_class_weights(
    labels,
    num_classes,
    *,
    mode="effective-number",
    beta=0.9999,
):
    """Return mean-one class weights without downsampling observations."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if len(counts) != num_classes or np.any(counts == 0):
        raise ValueError(
            "Every configured class must occur in the training split"
        )

    if mode == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    elif mode == "inverse-frequency":
        weights = counts.sum() / (num_classes * counts)
    elif mode == "effective-number":
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if beta == 0:
            weights = np.ones(num_classes, dtype=np.float64)
        else:
            weights = (1 - beta) / (1 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown class-weighting mode: {mode}")

    weights /= weights.mean()
    return weights.astype(np.float32)

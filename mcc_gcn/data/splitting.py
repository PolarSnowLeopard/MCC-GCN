"""Leakage-resistant molecular-pair split construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class GroupedSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    manifest: pd.DataFrame


def grouped_stratified_split(
    labels,
    pair_keys,
    *,
    validation_fraction=0.1,
    seed=42,
) -> GroupedSplit:
    """Split unordered pair groups before selecting augmented rows."""
    labels = np.asarray(labels, dtype=np.int64)
    pair_keys = np.asarray(pair_keys, dtype=str)
    if len(labels) != len(pair_keys):
        raise ValueError("labels and pair_keys differ in length")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if np.any(np.char.strip(pair_keys) == ""):
        raise ValueError("pair_keys must be non-empty")

    rows = pd.DataFrame(
        {
            "row_index": np.arange(len(labels), dtype=np.int64),
            "pair_key": pair_keys,
            "label_int": labels,
        }
    )
    label_counts_per_group = rows.groupby("pair_key")["label_int"].nunique()
    conflicting = label_counts_per_group[label_counts_per_group.gt(1)]
    if not conflicting.empty:
        raise ValueError(
            f"{len(conflicting)} pair groups contain conflicting labels"
        )

    groups = (
        rows.groupby("pair_key", sort=True)
        .agg(
            label_int=("label_int", "first"),
            augmented_row_count=("row_index", "size"),
        )
        .reset_index()
    )
    class_group_counts = groups["label_int"].value_counts()
    if class_group_counts.min() < 2:
        raise ValueError(
            "Every class needs at least two pair groups for stratified split"
        )

    validation_group_count = int(
        np.ceil(len(groups) * validation_fraction)
    )
    class_count = groups["label_int"].nunique()
    if validation_group_count < class_count:
        raise ValueError(
            "Validation split is too small to contain every class"
        )
    if len(groups) - validation_group_count < class_count:
        raise ValueError("Training split is too small to contain every class")

    train_keys, validation_keys = train_test_split(
        groups["pair_key"].to_numpy(),
        test_size=validation_fraction,
        random_state=seed,
        stratify=groups["label_int"].to_numpy(),
    )
    train_keys = set(train_keys)
    validation_keys = set(validation_keys)
    if train_keys.intersection(validation_keys):
        raise AssertionError("Pair-group overlap detected after split")

    split_by_key = {
        **{key: "train" for key in train_keys},
        **{key: "validation" for key in validation_keys},
    }
    groups["split"] = groups["pair_key"].map(split_by_key)
    groups["seed"] = seed
    groups["validation_fraction"] = validation_fraction
    groups = groups[
        [
            "pair_key",
            "label_int",
            "split",
            "augmented_row_count",
            "seed",
            "validation_fraction",
        ]
    ].sort_values(["split", "label_int", "pair_key"])

    row_splits = rows["pair_key"].map(split_by_key)
    train_indices = tuple(
        rows.loc[row_splits.eq("train"), "row_index"].astype(int)
    )
    validation_indices = tuple(
        rows.loc[row_splits.eq("validation"), "row_index"].astype(int)
    )
    return GroupedSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        manifest=groups.reset_index(drop=True),
    )

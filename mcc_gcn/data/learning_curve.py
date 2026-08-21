"""Deterministic physical-pair subsets for fine-tuning learning curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def balanced_nested_subsets(
    table: pd.DataFrame,
    sizes: list[int],
    *,
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Create class-balanced subsets where every smaller set is nested."""
    required = {
        "pair_key",
        "representative_pair_key",
        "label_str",
        "label_int",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Pair table is missing columns: {sorted(missing)}")
    if table["pair_key"].nunique() != len(table):
        raise ValueError("Pair table must contain unique physical pairs")
    if table["representative_pair_key"].nunique() != len(table):
        raise ValueError("Pair table must contain unique model-input pairs")

    sizes = sorted(set(int(size) for size in sizes))
    if not sizes or sizes[0] < 1:
        raise ValueError("At least one positive subset size is required")
    class_labels = sorted(table["label_int"].unique().tolist())
    class_count = len(class_labels)
    if any(size % class_count for size in sizes):
        raise ValueError("Every subset size must be divisible by class count")
    largest_per_class = sizes[-1] // class_count
    available = table["label_int"].value_counts()
    if any(
        available.get(label, 0) < largest_per_class
        for label in class_labels
    ):
        raise ValueError(
            "Largest balanced subset exceeds the smallest class"
        )

    rng = np.random.default_rng(seed)
    ranked_parts = []
    for label in class_labels:
        group = table.loc[table["label_int"].eq(label)].sort_values(
            "pair_key"
        )
        order = rng.permutation(len(group))
        ranked = group.iloc[order].copy().reset_index(drop=True)
        ranked["class_selection_rank"] = np.arange(1, len(ranked) + 1)
        ranked_parts.append(ranked)
    ranked_table = pd.concat(ranked_parts, ignore_index=True)

    subsets = {}
    for size in sizes:
        per_class = size // class_count
        subset = ranked_table.loc[
            ranked_table["class_selection_rank"].le(per_class)
        ].copy()
        if len(subset) != size:
            raise AssertionError("Balanced subset size mismatch")
        counts = subset["label_int"].value_counts()
        if counts.nunique() != 1 or counts.iloc[0] != per_class:
            raise AssertionError("Balanced subset class count mismatch")
        subset["subset_size"] = size
        subset["pairs_per_class"] = per_class
        subset["subset_seed"] = seed
        subset = subset.rename(
            columns={
                "pair_key": "physical_pair_key",
                "representative_pair_key": "model_pair_key",
            }
        )
        subsets[size] = subset.sort_values(
            ["label_int", "class_selection_rank", "physical_pair_key"]
        ).reset_index(drop=True)

    for smaller, larger in zip(sizes, sizes[1:]):
        small_keys = set(subsets[smaller]["physical_pair_key"])
        large_keys = set(subsets[larger]["physical_pair_key"])
        if not small_keys.issubset(large_keys):
            raise AssertionError("Learning-curve subsets are not nested")
    return subsets


def load_model_pair_keys(path: str | Path) -> set[str]:
    """Load and validate model pair keys from a frozen subset manifest."""
    table = pd.read_csv(path, keep_default_na=False)
    if "model_pair_key" not in table.columns:
        raise ValueError("Subset manifest is missing model_pair_key")
    keys = table["model_pair_key"].astype(str)
    if keys.str.strip().eq("").any():
        raise ValueError("Subset manifest contains empty model_pair_key")
    if keys.duplicated().any():
        raise ValueError("Subset manifest contains duplicate model_pair_key")
    return set(keys)


def filter_items_by_pair_keys(items, pair_keys: set[str]):
    """Keep all augmented rows belonging to the requested physical pairs."""
    available = {str(getattr(item, "pair_key", "")) for item in items}
    if "" in available:
        raise ValueError("Fine-tuning items are missing pair_key metadata")
    missing = pair_keys.difference(available)
    if missing:
        raise ValueError(
            f"Subset manifest contains {len(missing)} unavailable pairs"
        )
    selected = [
        item
        for item in items
        if str(getattr(item, "pair_key")) in pair_keys
    ]
    selected_keys = {str(item.pair_key) for item in selected}
    if selected_keys != pair_keys:
        raise AssertionError("Filtered fine-tuning pair set mismatch")
    return selected

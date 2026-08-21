"""Reconstruct the locked Minoxidil-only and KPX/KPR-64 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch_geometric.data import Data

from mcc_gcn.data.legacy import load_legacy_dense_items


@dataclass(frozen=True)
class MinoxidilOnlyData:
    minoxidil_items: tuple[Data, ...]
    external_ab_items: tuple[Data, ...]
    external_ba_items: tuple[Data, ...]
    minoxidil_pairs: pd.DataFrame
    external_pairs: pd.DataFrame


def _check_labels(items, expected, description):
    observed = np.asarray([int(item.y.item()) for item in items])
    expected = np.asarray(expected, dtype=np.int64)
    if not np.array_equal(observed, expected):
        mismatch = np.flatnonzero(observed != expected)
        raise ValueError(
            f"{description} labels disagree with the lock at rows "
            f"{mismatch[:5].tolist()}"
        )


def load_minoxidil_only_data(
    *,
    fine_tune_npz: str | Path,
    holdout_ab_npz: str | Path,
    holdout_ba_npz: str | Path,
    fine_tune_manifest: str | Path,
    external_manifest: str | Path,
    preserve_padding: bool,
) -> MinoxidilOnlyData:
    """Recover 20 Minoxidil train pairs and all 64 protected target pairs."""
    fine_tune_lock = pd.read_csv(fine_tune_manifest, keep_default_na=False)
    minoxidil = (
        fine_tune_lock.loc[fine_tune_lock["source_dataset"].eq("minoxidil_20")]
        .sort_values("source_index")
        .reset_index(drop=True)
    )
    if len(minoxidil) != 20:
        raise ValueError(f"Expected 20 locked Minoxidil pairs, found {len(minoxidil)}")
    if minoxidil["source_index"].astype(int).tolist() != list(range(20)):
        raise ValueError("Minoxidil source indices are not the frozen 0..19 sequence")

    minoxidil_keys = minoxidil["lock_id"].astype(str).tolist()
    minoxidil_ab = load_legacy_dense_items(
        fine_tune_npz,
        row_indices=range(20),
        pair_keys=minoxidil_keys,
        preserve_padding=preserve_padding,
    )
    minoxidil_ba = load_legacy_dense_items(
        fine_tune_npz,
        row_indices=range(20, 40),
        pair_keys=minoxidil_keys,
        preserve_padding=preserve_padding,
    )
    expected_minoxidil_labels = minoxidil["label_int"].astype(int).to_numpy()
    _check_labels(minoxidil_ab, expected_minoxidil_labels, "Minoxidil A/B")
    _check_labels(minoxidil_ba, expected_minoxidil_labels, "Minoxidil B/A")

    external = pd.read_csv(external_manifest, keep_default_na=False)
    external = external.sort_values("external_index").reset_index(drop=True)
    if len(external) != 64:
        raise ValueError(f"Expected 64 locked external pairs, found {len(external)}")
    if external["external_index"].astype(int).tolist() != list(range(64)):
        raise ValueError("External pair indices are not the frozen 0..63 sequence")

    holdout = external.loc[external["split"].eq("holdout")].copy()
    target_ft = external.loc[external["split"].eq("fine_tune")].copy()
    if len(holdout) != 50 or len(target_ft) != 14:
        raise ValueError(
            "External split must contain 50 holdout and 14 former fine-tuning pairs"
        )

    holdout_keys = [f"external-{int(value):02d}" for value in holdout["external_index"]]
    holdout_rows = holdout["holdout_feature_index"].astype(int).tolist()
    holdout_ab = load_legacy_dense_items(
        holdout_ab_npz,
        row_indices=holdout_rows,
        pair_keys=holdout_keys,
        preserve_padding=preserve_padding,
    )
    holdout_ba = load_legacy_dense_items(
        holdout_ba_npz,
        row_indices=holdout_rows,
        pair_keys=holdout_keys,
        preserve_padding=preserve_padding,
    )
    expected_holdout_labels = holdout["label_int"].astype(int).to_numpy()
    _check_labels(holdout_ab, expected_holdout_labels, "KPX/KPR holdout A/B")
    _check_labels(holdout_ba, expected_holdout_labels, "KPX/KPR holdout B/A")

    target_keys = [
        f"external-{int(value):02d}" for value in target_ft["external_index"]
    ]
    target_ab = load_legacy_dense_items(
        fine_tune_npz,
        row_indices=target_ft["fine_tune_forward_feature_index"].astype(int),
        pair_keys=target_keys,
        preserve_padding=preserve_padding,
    )
    target_ba = load_legacy_dense_items(
        fine_tune_npz,
        row_indices=target_ft["fine_tune_reverse_feature_index"].astype(int),
        pair_keys=target_keys,
        preserve_padding=preserve_padding,
    )
    expected_target_labels = target_ft["label_int"].astype(int).to_numpy()
    _check_labels(target_ab, expected_target_labels, "KPX/KPR former FT A/B")
    _check_labels(target_ba, expected_target_labels, "KPX/KPR former FT B/A")

    ab_by_key = {item.pair_key: item for item in [*holdout_ab, *target_ab]}
    ba_by_key = {item.pair_key: item for item in [*holdout_ba, *target_ba]}
    all_keys = [f"external-{index:02d}" for index in range(64)]
    if set(ab_by_key) != set(all_keys) or set(ba_by_key) != set(all_keys):
        raise ValueError("Reconstructed KPX/KPR-64 pair keys are incomplete")
    external_ab = [ab_by_key[key] for key in all_keys]
    external_ba = [ba_by_key[key] for key in all_keys]
    expected_external_labels = external["label_int"].astype(int).to_numpy()
    _check_labels(external_ab, expected_external_labels, "KPX/KPR-64 A/B")
    _check_labels(external_ba, expected_external_labels, "KPX/KPR-64 B/A")

    return MinoxidilOnlyData(
        minoxidil_items=(*minoxidil_ab, *minoxidil_ba),
        external_ab_items=tuple(external_ab),
        external_ba_items=tuple(external_ba),
        minoxidil_pairs=minoxidil,
        external_pairs=external,
    )

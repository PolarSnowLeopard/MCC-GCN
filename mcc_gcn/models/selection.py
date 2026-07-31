"""Validation checkpoint selection helpers."""

from __future__ import annotations

import math


def validation_result_improved(
    metric,
    loss,
    best_metric,
    best_loss,
    *,
    metric_tolerance=1e-12,
):
    """Prefer a higher metric, then a lower loss when metrics are tied."""
    if metric > best_metric + metric_tolerance:
        return True
    return math.isclose(
        metric,
        best_metric,
        rel_tol=0.0,
        abs_tol=metric_tolerance,
    ) and loss < best_loss

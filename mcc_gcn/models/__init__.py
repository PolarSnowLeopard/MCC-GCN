"""Model and metric entry points with optional Torch dependencies."""

__all__ = [
    "GCNNet",
    "train_epoch",
    "evaluate",
    "calculate_metrics",
    "calculate_detailed_metrics",
]


def __getattr__(name):
    if name in {"GCNNet", "train_epoch", "evaluate"}:
        from .gcn import GCNNet, evaluate, train_epoch

        return {
            "GCNNet": GCNNet,
            "train_epoch": train_epoch,
            "evaluate": evaluate,
        }[name]
    if name in {"calculate_metrics", "calculate_detailed_metrics"}:
        from .metrics import calculate_detailed_metrics, calculate_metrics

        return {
            "calculate_metrics": calculate_metrics,
            "calculate_detailed_metrics": calculate_detailed_metrics,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

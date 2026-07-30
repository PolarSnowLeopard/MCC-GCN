"""Data loading and quality-control utilities."""

__all__ = ["GraphDataLoader", "GraphDataset"]


def __getattr__(name):
    if name in __all__:
        from .dataset import GraphDataLoader, GraphDataset

        return {
            "GraphDataset": GraphDataset,
            "GraphDataLoader": GraphDataLoader,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

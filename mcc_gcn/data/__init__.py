from .dataset import GraphDataset, GraphDataLoader


def __getattr__(name):
    if name == "DataFilter":
        from .filter import DataFilter
        return DataFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

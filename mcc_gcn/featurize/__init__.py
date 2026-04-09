from .bond import Bond
from .cocrystal import Cocrystal
from .fingerprint import Fingerprint
from .vertex_matrix import VertexMatrix
from .adjacent_tensor import AdjacentTensor
from .descriptors import compute_descriptors
from .rdkit_coformer import RDKitCoformer

_CCDC_LAZY = {
    'change_hbond_criterion': '.hbond',
    'Atom': '.atom',
    'Coformer': '.coformer',
}


def __getattr__(name):
    if name in _CCDC_LAZY:
        import importlib
        mod = importlib.import_module(_CCDC_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

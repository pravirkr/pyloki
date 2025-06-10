try:
    import cupy as cp  # noqa: F401
except ImportError as e:
    msg = (
        "pyloki[cuda] is not installed or no CUDA toolkit found.  "
        "Install with `pip install pyloki[cuda]` and ensure CUDA is on your PATH."
    )
    raise ImportError(msg) from e

# if import succeeds, expose your GPU API
from .kernels import brutefold_start_cuda

__all__ = [
    "brutefold_start_cuda",
]

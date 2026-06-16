"""Unified array backend: CuPy when a CUDA device is available, NumPy otherwise."""
import os

_DEVICE_PREFERENCE = os.environ.get("NIMBLEML_DEVICE", "auto").strip().lower()


def _try_cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except (ImportError, Exception):
        pass
    return None


def _init_backend(preference):
    if preference == "cpu":
        import numpy as np

        return np, False, "cpu"

    if preference in ("gpu", "cuda"):
        cp = _try_cupy()
        if cp is None:
            raise RuntimeError(
                "NIMBLEML_DEVICE=gpu but CuPy is unavailable or no CUDA device was found."
            )
        return cp, True, "gpu"

    cp = _try_cupy()
    if cp is not None:
        return cp, True, "gpu"

    import numpy as np

    return np, False, "cpu"


np, using_gpu, device = _init_backend(_DEVICE_PREFERENCE)


_DTYPE_PREFERENCE = os.environ.get("NIMBLEML_DTYPE", "float32").strip().lower()


def _resolve_dtype(name):
    name = name.strip().lower()
    if name in ("float32", "f32", "single"):
        return np.float32
    if name in ("float64", "f64", "double"):
        return np.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


# Global compute dtype. float32 is ~2x faster on CPU and much faster on GPU,
# while using half the memory; float64 is available for strict gradchecks.
dtype = _resolve_dtype(_DTYPE_PREFERENCE)


def get_dtype():
    """Return the current global compute dtype."""
    return dtype


def set_dtype(name):
    """Set the global compute dtype ('float32' or 'float64').

    Modules read this dynamically, so it may be changed at runtime (e.g. tests
    pin float64 for finite-difference gradchecks).
    """
    global dtype, _DTYPE_PREFERENCE
    resolved = _resolve_dtype(name)
    _DTYPE_PREFERENCE = "float64" if resolved == np.float64 else "float32"
    os.environ["NIMBLEML_DTYPE"] = _DTYPE_PREFERENCE
    dtype = resolved
    return dtype


def set_device(name):
    """Select the array backend ('auto', 'cpu', or 'gpu').

    Must be called before importing other NimbleML modules, or set
    NIMBLEML_DEVICE in the environment before starting Python.
    """
    global np, using_gpu, device, _DEVICE_PREFERENCE, dtype

    name = name.strip().lower()
    if name not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")

    preference = "gpu" if name == "cuda" else name
    _DEVICE_PREFERENCE = preference
    os.environ["NIMBLEML_DEVICE"] = preference
    np, using_gpu, device = _init_backend(preference)
    dtype = _resolve_dtype(_DTYPE_PREFERENCE)

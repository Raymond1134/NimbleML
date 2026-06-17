"""Unified array backend: CuPy when a CUDA device is available, NumPy otherwise."""
from __future__ import annotations

import os
from typing import Optional

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
    if name in ("float16", "f16", "half", "bfloat16", "bf16"):
        raise ValueError(
            "fp16/bf16 are not supported yet; use float32 on GPU until the "
            "float32 path matches a PyTorch baseline."
        )
    if name in ("float32", "f32", "single"):
        return np.float32
    if name in ("float64", "f64", "double"):
        return np.float64
    raise ValueError("dtype must be 'float32' or 'float64'")


# Global compute dtype. float32 is the training default on GPU; float64 is for
# gradcheck / strict CPU tests only. fp16/bf16 are intentionally unsupported
# until the float32 GPU path matches a PyTorch baseline.
dtype = _resolve_dtype(_DTYPE_PREFERENCE)


def as_int64(data):
    """Array of token indices on the active backend (CPU NumPy or GPU CuPy)."""
    return np.asarray(data, dtype=np.int64)


def apply_runtime_config(device: Optional[str] = None, dtype_name: Optional[str] = None):
    """Apply device/dtype after ``NIMBLEML_*`` env vars or explicit arguments."""
    if device is not None:
        set_device(device)
    elif os.environ.get("NIMBLEML_DEVICE"):
        set_device(os.environ["NIMBLEML_DEVICE"])

    if dtype_name is not None:
        set_dtype(dtype_name)
    elif os.environ.get("NIMBLEML_DTYPE"):
        set_dtype(os.environ["NIMBLEML_DTYPE"])


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

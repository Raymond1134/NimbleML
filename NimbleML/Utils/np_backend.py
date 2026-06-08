# np_backend.py
# Unified array backend: CuPy when a CUDA device is available, NumPy otherwise.
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


def set_device(name):
    """Select the array backend ('auto', 'cpu', or 'gpu').

    Must be called before importing other NimbleML modules, or set
    NIMBLEML_DEVICE in the environment before starting Python.
    """
    global np, using_gpu, device, _DEVICE_PREFERENCE

    name = name.strip().lower()
    if name not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")

    preference = "gpu" if name == "cuda" else name
    _DEVICE_PREFERENCE = preference
    os.environ["NIMBLEML_DEVICE"] = preference
    np, using_gpu, device = _init_backend(preference)

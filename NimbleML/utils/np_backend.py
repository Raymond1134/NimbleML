"""Unified array backend: CuPy when a CUDA device is available, NumPy otherwise."""
from __future__ import annotations
import os
from typing import Optional

_DEVICE_PREFERENCE = os.environ.get("NIMBLEML_DEVICE", "auto").strip().lower()

# Populated by configure_gpu_runtime() on the first GPU init.
_gpu_runtime_configured = False
_gpu_runtime_info: dict = {}
_device_stream = None
_memory_pool = None
_pinned_pool = None


def _try_cupy():
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except (ImportError, Exception):
        pass
    return None


def configure_gpu_runtime(*, verbose: bool = False) -> dict:
    """Enable TF32, caching allocator, and a default CUDA stream for CuPy.

    Idempotent. No-op when the backend is CPU. Returns a small status dict
    (device name, TF32, pool) useful for training logs.
    """
    global _gpu_runtime_configured, _gpu_runtime_info, _device_stream
    global _memory_pool, _pinned_pool

    if not using_gpu:
        _gpu_runtime_info = {"device": "cpu", "tf32": False, "pool": False}
        if verbose:
            print("[gpu] cpu backend", flush=True)
        return _gpu_runtime_info
    if _gpu_runtime_configured:
        if verbose:
            info = _gpu_runtime_info
            print(
                f"[gpu] {info.get('name', 'cuda')} tf32={info.get('tf32')} "
                f"mempool={info.get('pool')}",
                flush=True,
            )
        return _gpu_runtime_info

    import cupy as cp

    info: dict = {"device": "gpu", "tf32": False, "pool": False, "name": ""}
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get("name")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        info["name"] = name or f"cuda:{cp.cuda.Device().id}"
    except Exception:
        info["name"] = f"cuda:{cp.cuda.Device().id}"

    # TF32 for fp32 GEMMs on Ampere+ (Ada L40S included). fp16 already uses tensor cores.
    # CUPY_TF32 is read by CuPy's cuBLAS/cuDNN bindings; also try in-process hooks.
    os.environ.setdefault("CUPY_TF32", "1")
    tf32_on = os.environ.get("CUPY_TF32", "1").strip() not in ("0", "false", "off")
    info["tf32"] = tf32_on
    try:
        # CuPy 13+: allow_tf32 on the CUB module when present.
        cub = getattr(cp.cuda, "cub", None)
        if cub is not None and hasattr(cub, "set_allow_tf32"):
            cub.set_allow_tf32(True)
    except Exception:
        pass
    try:
        cudnn = getattr(cp, "cudnn", None) or getattr(cp.cuda, "cudnn", None)
        if cudnn is not None and hasattr(cudnn, "set_allow_tf32"):
            cudnn.set_allow_tf32(True)
    except Exception:
        pass

    # Caching memory allocator (CuPy default pool is fine; pin it explicitly).
    try:
        _memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(_memory_pool.malloc)
        _pinned_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(_pinned_pool.malloc)
        info["pool"] = True
    except Exception:
        info["pool"] = False

    try:
        _device_stream = cp.cuda.Stream(null=False, non_blocking=True)
    except Exception:
        _device_stream = None

    _gpu_runtime_configured = True
    _gpu_runtime_info = info
    if verbose:
        print(
            f"[gpu] {info.get('name', 'cuda')} tf32={info['tf32']} "
            f"mempool={info['pool']}",
            flush=True,
        )
    return info


def gpu_runtime_info() -> dict:
    """Return the last configure_gpu_runtime() status (may be empty)."""
    return dict(_gpu_runtime_info)


def device_stream():
    """Non-blocking CUDA stream created by configure_gpu_runtime(), or None."""
    return _device_stream


def sync_device() -> None:
    """Block until outstanding GPU work finishes (no-op on CPU)."""
    if not using_gpu:
        return
    import cupy as cp

    cp.cuda.Device().synchronize()
    if _device_stream is not None:
        _device_stream.synchronize()


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
if using_gpu:
    configure_gpu_runtime(verbose=False)


_DTYPE_PREFERENCE = os.environ.get("NIMBLEML_DTYPE", "float32").strip().lower()


def _resolve_dtype(name):
    name = name.strip().lower()
    if name in ("float16", "f16", "half"):
        return np.float16
    if name in ("bfloat16", "bf16"):
        # Prefer native bf16 when the backend exposes it; else float16.
        return getattr(np, "bfloat16", np.float16)
    if name in ("float32", "f32", "single"):
        return np.float32
    if name in ("float64", "f64", "double"):
        return np.float64
    raise ValueError("dtype must be 'float16', 'bfloat16', 'float32', or 'float64'")


# Global compute dtype. float32 is the training default; fp16/bf16 require the
# native extension and GradScaler for stable training.
dtype = _resolve_dtype(_DTYPE_PREFERENCE)


def as_int64(data):
    """Array of token indices on the active backend (CPU NumPy or GPU CuPy)."""
    return np.asarray(data, dtype=np.int64)


def on_device(arr, *, dtype=None):
    """Copy *arr* onto the active backend in a contiguous buffer.

    Shared by the fused kernels so they do not each redefine this helper.
    """
    kwargs = {} if dtype is None else {"dtype": dtype}
    return np.ascontiguousarray(np.asarray(arr, **kwargs))


def as_label_indices(label_indices, *, batch_size: int):
    """Validate and flatten int64 class labels for cross-entropy kernels.

    Accepts a ``Tensor`` (duck-typed: has ``.data`` but no ``.dtype``), a
    list/tuple, or a NumPy/CuPy ndarray.
    """
    # ndarrays (numpy/cupy) expose ``.dtype``; a NimbleML Tensor does not, so
    # unwrap it. NumPy/CuPy ``.data`` is a raw buffer and must NOT be unwrapped.
    if not hasattr(label_indices, "dtype") and hasattr(label_indices, "data"):
        label_indices = label_indices.data
    labels = np.asarray(label_indices, dtype=np.int64).reshape(-1)
    if labels.size != batch_size:
        raise ValueError(
            f"label count ({labels.size}) must match batch size ({batch_size})."
        )
    return labels


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
    if resolved == np.float64:
        _DTYPE_PREFERENCE = "float64"
    elif resolved == getattr(np, "bfloat16", None):
        _DTYPE_PREFERENCE = "bfloat16"
    elif resolved == np.float16:
        _DTYPE_PREFERENCE = "float16"
    else:
        _DTYPE_PREFERENCE = "float32"
    os.environ["NIMBLEML_DTYPE"] = _DTYPE_PREFERENCE
    dtype = resolved
    return dtype


def set_device(name):
    """Select the array backend ('auto', 'cpu', or 'gpu').

    Must be called before importing other NimbleML modules, or set
    NIMBLEML_DEVICE in the environment before starting Python.
    """
    global np, using_gpu, device, _DEVICE_PREFERENCE, dtype
    global _gpu_runtime_configured, _gpu_runtime_info, _device_stream
    global _memory_pool, _pinned_pool

    name = name.strip().lower()
    if name not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")

    preference = "gpu" if name == "cuda" else name
    _DEVICE_PREFERENCE = preference
    os.environ["NIMBLEML_DEVICE"] = preference
    np, using_gpu, device = _init_backend(preference)
    dtype = _resolve_dtype(_DTYPE_PREFERENCE)
    _gpu_runtime_configured = False
    _gpu_runtime_info = {}
    _device_stream = None
    _memory_pool = None
    _pinned_pool = None
    if using_gpu:
        configure_gpu_runtime(verbose=False)

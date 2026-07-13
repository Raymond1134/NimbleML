"""Bridge NumPy/CuPy arrays to the required ``nimbleml_native`` extension.

Only the GELU CPU fast path still routes through here; the other kernels run
directly on the array backend (see ``NimbleML/kernels/``).
"""
from __future__ import annotations

import numpy as host_np

from NimbleML._native_loader import native
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, on_device, using_gpu


def _to_host_f32(arr) -> host_np.ndarray:
    a = on_device(arr, dtype=np_backend.dtype)
    get = getattr(a, "get", None)
    host = host_np.asarray(get() if get is not None else a, dtype=host_np.float32)
    return host_np.ascontiguousarray(host)


def _from_host(host_arr):
    """Place a host float32 array onto the active backend."""
    return np.asarray(host_arr, dtype=np_backend.dtype)


def native_gelu_forward(arr):
    x = _to_host_f32(arr).reshape(-1)
    out, tanh_u = native.gelu_forward(x)
    shape = on_device(arr, dtype=np_backend.dtype).shape
    return _from_host(out).reshape(shape), _from_host(tanh_u).reshape(shape)


def native_gelu_backward(grad_out, arr, tanh_u):
    g = _to_host_f32(grad_out).reshape(-1)
    x = _to_host_f32(arr).reshape(-1)
    t = _to_host_f32(tanh_u).reshape(-1)
    gx = native.gelu_backward(g, x, t)
    return _from_host(gx).reshape(on_device(arr, dtype=np_backend.dtype).shape)


__all__ = [
    "native",
    "native_gelu_forward",
    "native_gelu_backward",
    "using_gpu",
]

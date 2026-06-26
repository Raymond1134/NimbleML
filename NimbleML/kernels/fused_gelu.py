"""Fused GELU forward and backward on the active NumPy/CuPy backend.
The forward pass caches ``tanh_u`` so backward does not recompute ``tanh(u)``.
"""
from __future__ import annotations
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np

_GELU_K = np.sqrt(2.0 / np.pi)
_GELU_COEF = 0.044715
_DU_DX_COEF = 0.134145


def _on_device(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np_backend.dtype))


def fused_gelu_forward(arr):
    """Apply GELU element-wise.

    Args:
        arr (array-like): Input activations.

    Returns:
        tuple[ndarray, ndarray]:
            - out: GELU output, same shape as ``arr``.
            - tanh_u: Cached ``tanh(u)`` values for :func:`fused_gelu_backward`.
    """
    x = _on_device(arr)
    x3 = x * x * x
    tanh_u = np.tanh(_GELU_K * (x + _GELU_COEF * x3))
    out = 0.5 * x * (1.0 + tanh_u)
    return out, tanh_u


def fused_gelu_backward(grad_out, arr, tanh_u=None):
    """Backpropagate through :func:`fused_gelu_forward`.

    Args:
        grad_out (array-like): Upstream gradient, same shape as forward input.
        arr (array-like): Forward input activations.
        tanh_u (array-like, optional): Cached values from forward. Recomputed
            when omitted.

    Returns:
        ndarray: Gradient with respect to ``arr``.
    """
    grad = _on_device(grad_out)
    x = _on_device(arr)
    if tanh_u is None:
        x3 = x * x * x
        tanh_u = np.tanh(_GELU_K * (x + _GELU_COEF * x3))
    else:
        tanh_u = _on_device(tanh_u)

    du_dx = _GELU_K * (1.0 + _DU_DX_COEF * x * x)
    sech2 = 1.0 - tanh_u * tanh_u
    grad_x = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du_dx
    return grad * grad_x

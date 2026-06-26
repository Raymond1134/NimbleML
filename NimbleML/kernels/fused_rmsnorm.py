"""Fused RMSNorm forward and backward on the active NumPy/CuPy backend.

RMSNorm over the last dimension:

    y = gamma * x / sqrt(mean(x^2) + epsilon)
"""
from __future__ import annotations
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np


def _on_device(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np_backend.dtype))


def fused_rmsnorm_forward(x, gamma, epsilon=1e-5):
    """Apply RMS normalization along the last axis.

    Args:
        x (array-like): Input of shape ``(..., normalized_dim)``.
        gamma (array-like): Scale vector of shape ``(normalized_dim,)``.
        epsilon (float): Stability constant added inside the square root.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            - out: Normalized output, same shape as ``x``.
            - x: Contiguous input buffer used for backward.
            - ms: Mean square per row, shape ``(..., 1)``.
            - rms: Root mean square per row, shape ``(..., 1)``.
    """
    x_arr = _on_device(x)
    g_arr = _on_device(gamma)
    ms = np.mean(x_arr * x_arr, axis=-1, keepdims=True)
    rms = np.sqrt(ms + epsilon)
    out = (x_arr / rms) * g_arr
    return out, x_arr, ms, rms


def fused_rmsnorm_backward(grad_out, x, gamma, ms, rms, epsilon=1e-5):
    """Backpropagate through :func:`fused_rmsnorm_forward`.

    Args:
        grad_out (array-like): Upstream gradient, same shape as forward output.
        x (array-like): Forward input buffer.
        gamma (array-like): Scale vector from forward.
        ms (array-like): Cached mean-square values from forward.
        rms (array-like): Cached RMS values from forward.
        epsilon (float): Same epsilon used in forward.

    Returns:
        tuple[ndarray, ndarray]:
            - grad_x: Gradient with respect to ``x``.
            - grad_gamma: Gradient with respect to ``gamma``.
    """
    grad = _on_device(grad_out)
    x_arr = _on_device(x)
    g_arr = _on_device(gamma)
    ms_arr = _on_device(ms)
    rms_arr = _on_device(rms)
    d = x_arr.shape[-1]

    x_hat = x_arr / rms_arr
    grad_gamma = np.sum(grad * x_hat, axis=0)

    grad_x_hat = grad * g_arr
    grad_ms = np.sum(
        grad_x_hat * x_arr * (-0.5) * (ms_arr + epsilon) ** (-1.5),
        axis=-1,
        keepdims=True,
    )
    grad_x_from_ms = (2.0 / d) * x_arr * grad_ms
    grad_x_direct = grad_x_hat / rms_arr
    grad_x = grad_x_direct + grad_x_from_ms
    return grad_x, grad_gamma

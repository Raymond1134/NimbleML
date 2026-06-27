"""Fused tied LM head + cross-entropy (hidden @ W^T then CE)."""
from __future__ import annotations
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np


def _on_device(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np_backend.dtype))


def _project_logits(hidden, weights):
    w = _on_device(weights)
    h = _on_device(hidden)
    w_t = np.ascontiguousarray(np.swapaxes(w, -2, -1))
    return np.matmul(h, w_t)


def fused_tied_crossentropy_forward(hidden, weights, label_indices):
    """Mean CE from hidden states and tied embedding weights.

    Args:
        hidden (array-like): Shape ``(batch, d_model)``.
        weights (array-like): Tied embedding matrix ``(vocab, d_model)``.
        label_indices (array-like): int64 labels ``(batch,)``.

    Returns:
        tuple: ``(loss, hidden, weights, max_vals, sum_exp)`` for backward.
    """
    h = _on_device(hidden)
    w = _on_device(weights)
    logits = _project_logits(h, w)
    loss, _, max_vals, sum_exp = fused_crossentropy_forward(logits, label_indices)
    return loss, h, w, max_vals, sum_exp


def fused_tied_crossentropy_backward(
    grad_scale,
    hidden,
    weights,
    label_indices,
    max_vals,
    sum_exp,
):
    """Gradients w.r.t. hidden and tied weights from fused tied CE.

    Returns:
        tuple[ndarray, ndarray]: ``(grad_hidden, grad_weights)``.
    """
    h = _on_device(hidden)
    w = _on_device(weights)
    logits = _project_logits(h, w)
    grad_logits = fused_crossentropy_backward(
        grad_scale,
        logits,
        label_indices,
        max_vals,
        sum_exp,
    )
    grad_h = np.matmul(grad_logits, w)
    grad_w = np.matmul(np.ascontiguousarray(np.swapaxes(grad_logits, -2, -1)), h)
    return grad_h, grad_w

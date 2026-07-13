"""Fused tied LM head + cross-entropy (hidden @ W^T then CE)."""
from __future__ import annotations
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, on_device


def fused_tied_crossentropy_forward(hidden, weights, label_indices):
    """Mean CE from hidden states and tied embedding weights.

    Args:
        hidden (array-like): Shape ``(batch, d_model)``.
        weights (array-like): Tied embedding matrix ``(vocab, d_model)``.
        label_indices (array-like): int64 labels ``(batch,)``.

    Returns:
        tuple: ``(loss, hidden, weights, logits, max_vals, sum_exp)`` for backward.
    """
    h = on_device(hidden, dtype=np_backend.dtype)
    w = on_device(weights, dtype=np_backend.dtype)
    # Transposed view — BLAS/cuBLAS take strided operands, no copy needed.
    logits = np.matmul(h, np.swapaxes(w, -2, -1))
    loss, logits_arr, max_vals, sum_exp = fused_crossentropy_forward(logits, label_indices)
    return loss, h, w, logits_arr, max_vals, sum_exp


def fused_tied_crossentropy_backward(
    grad_scale,
    hidden,
    weights,
    label_indices,
    logits,
    max_vals,
    sum_exp,
):
    """Gradients w.r.t. hidden and tied weights from fused tied CE.

    Args:
        logits: Logits buffer saved from ``fused_tied_crossentropy_forward``;
            avoids recomputing the vocab-sized ``hidden @ W^T`` GEMM.

    Returns:
        tuple[ndarray, ndarray]: ``(grad_hidden, grad_weights)``.
    """
    h = on_device(hidden, dtype=np_backend.dtype)
    w = on_device(weights, dtype=np_backend.dtype)
    grad_logits = fused_crossentropy_backward(
        grad_scale,
        logits,
        label_indices,
        max_vals,
        sum_exp,
    )
    grad_h = np.matmul(grad_logits, w)
    grad_w = np.matmul(np.swapaxes(grad_logits, -2, -1), h)
    return grad_h, grad_w

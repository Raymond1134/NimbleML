"""Fused log-softmax + cross-entropy on the active NumPy/CuPy backend.

Forward computes mean negative log-likelihood without materializing full
softmax probabilities. Backward recomputes probabilities from cached
``max_vals`` and ``sum_exp`` instead of storing the full ``(N, C)`` prob
tensor from forward.
"""
from __future__ import annotations
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np


def _on_device(arr):
    return np.ascontiguousarray(np.asarray(arr, dtype=np_backend.dtype))


def _label_indices(label_indices, *, batch_size: int):
    labels = np.asarray(label_indices, dtype=np.int64).reshape(-1)
    if labels.size != batch_size:
        raise ValueError(
            f"label count ({labels.size}) must match batch size ({batch_size})."
        )
    return labels


def fused_crossentropy_forward(logits, label_indices):
    """Mean cross-entropy from raw logits and integer class labels.

    Args:
        logits (array-like): Logits of shape ``(batch, classes)``.
        label_indices (array-like): int64 labels of shape ``(batch,)``.

    Returns:
        tuple[float, ndarray, ndarray, ndarray]:
            - loss: Scalar mean cross-entropy.
            - logits: Contiguous logits buffer used for backward.
            - max_vals: Per-row max for numerical stability, shape ``(batch, 1)``.
            - sum_exp: Per-row sum of ``exp(logits - max)``, shape ``(batch, 1)``.
    """
    logits_arr = _on_device(logits)
    batch_size, _ = logits_arr.shape
    labels = _label_indices(label_indices, batch_size=batch_size)

    max_vals = np.max(logits_arr, axis=1, keepdims=True)
    shifted = logits_arr - max_vals
    sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True)
    log_sum_exp = max_vals.ravel() + np.log(sum_exp.ravel())

    row_idx = np.arange(batch_size, dtype=np.int64)
    correct_logits = logits_arr[row_idx, labels]
    per_sample = log_sum_exp - correct_logits
    loss = float(np.sum(per_sample) / batch_size)
    return loss, logits_arr, max_vals, sum_exp


def fused_crossentropy_backward(
    grad_scale,
    logits,
    label_indices,
    max_vals,
    sum_exp,
):
    """Gradient of logits for mean cross-entropy.

    Args:
        grad_scale (float): Upstream gradient of the scalar loss.
        logits (array-like): Forward logits buffer, shape ``(batch, classes)``.
        label_indices (array-like): int64 labels used in forward.
        max_vals (array-like): Cached row max values from forward.
        sum_exp (array-like): Cached row normalizers from forward.

    Returns:
        ndarray: Gradient with respect to ``logits``, shape ``(batch, classes)``.
    """
    logits_arr = _on_device(logits)
    batch_size, _ = logits_arr.shape
    labels = _label_indices(label_indices, batch_size=batch_size)
    max_arr = _on_device(max_vals)
    sum_arr = _on_device(sum_exp)

    shifted = logits_arr - max_arr
    probs = np.exp(shifted) / sum_arr
    grad = probs.copy()
    row_idx = np.arange(batch_size, dtype=np.int64)
    grad[row_idx, labels] -= 1.0
    grad /= batch_size
    return grad * float(grad_scale)

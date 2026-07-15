"""Shared logits / label preparation for cross-entropy losses."""
from __future__ import annotations
import numpy as host_np
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_label_indices, np
from NimbleML.utils.tensor import Tensor


def flatten_logits(logits: Tensor):
    """Reshape 1D/2D/3D logits to ``(rows, classes)``.

    Returns:
        logits_arr, total_batch, class_count, out_shape
    """
    out_shape = logits.shape
    if logits.ndim == 1:
        total_batch = 1
        class_count = logits.shape[0]
        logits_arr = logits._view((1, class_count))
    elif logits.ndim == 2:
        total_batch, class_count = logits.shape
        logits_arr = logits._view((total_batch, class_count))
    elif logits.ndim == 3:
        total_batch = logits.shape[0] * logits.shape[1]
        class_count = logits.shape[2]
        logits_arr = logits._view((total_batch, class_count))
    else:
        raise ValueError("Cross-entropy expects 1D, 2D, or 3D logits.")
    return logits_arr, total_batch, class_count, out_shape


def labels_to_device(labels, batch_size: int):
    """Validate labels and return an int64 vector on the active backend."""
    return as_label_indices(labels, batch_size=batch_size)


def labels_to_host(labels, batch_size: int) -> host_np.ndarray:
    """Deprecated alias: host copy of :func:`labels_to_device` (tests / CPU tooling)."""
    label_indices = labels_to_device(labels, batch_size=batch_size)
    return host_np.asarray(
        label_indices.get() if hasattr(label_indices, "get") else label_indices,
        dtype=host_np.int64,
    ).copy()


def filter_ignore_index(logits_arr, label_indices, ignore_index, *, empty_requires_grad: bool):
    """Drop rows whose label equals *ignore_index* (works on NumPy or CuPy).

    Returns:
        ``(logits_arr, label_indices, valid_mask, empty_loss)``.
        When every row is ignored, ``empty_loss`` is a zero scalar Tensor and
        the other fields are unused. Otherwise ``empty_loss`` is ``None``.
    """
    total_batch = int(label_indices.size)
    if ignore_index is None:
        valid = np.ones(total_batch, dtype=bool)
        return logits_arr, label_indices, valid, None

    valid = label_indices != ignore_index
    # ``np.any`` on CuPy returns a 0-d device array — cast via bool().
    any_valid = bool(np.any(valid))
    if not any_valid:
        return (
            logits_arr,
            label_indices,
            valid,
            Tensor([0.0], (), requires_grad=empty_requires_grad),
        )
    return logits_arr[valid], label_indices[valid], valid, None


def scatter_row_grads(grad, valid_mask, *, total_batch: int, class_count: int, ignore_index):
    """Expand filtered row grads back to the full ``(total_batch, class_count)`` layout."""
    if ignore_index is None:
        return grad
    full_grad = np.zeros((total_batch, class_count), dtype=np_backend.dtype)
    full_grad[valid_mask] = grad
    return full_grad


def reshape_grad_to_logits(full_grad, logits: Tensor, out_shape: tuple):
    """Restore original logits layout for 1D/3D inputs."""
    if logits.ndim in (1, 3):
        return full_grad.reshape(out_shape)
    return full_grad

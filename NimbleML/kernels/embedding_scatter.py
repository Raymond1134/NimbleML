"""Embedding gather (forward) and scatter-add (backward)."""
from __future__ import annotations
import numpy as host_np
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np


def _on_device(arr, *, dtype=None):
    """Copy *arr* onto the active backend in a contiguous buffer."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    return np.ascontiguousarray(np.asarray(arr, **kwargs))


def embedding_lookup(weights, ids):
    """Gather embedding rows (forward pass).

    Args:
        weights (array-like): Embedding matrix of shape ``(vocab_size, embed_dim)``.
        ids (array-like): Integer token IDs of shape ``(n,)`` or any shape that
            will be flattened for indexing.

    Returns:
        ndarray: Gathered embeddings of shape ``(*ids.shape, embed_dim)`` when
            ``ids`` is multidimensional, or ``(n, embed_dim)`` when flat.
    
    Raises:
        ValueError: If the token IDs are out of range [0, vocab_size).
    """
    w = _on_device(weights, dtype=np_backend.dtype)
    vocab_size, embed_dim = w.shape
    flat_ids = _on_device(ids, dtype=np.int64).reshape(-1)
    if flat_ids.size and (int(flat_ids.min()) < 0 or int(flat_ids.max()) >= vocab_size):
        raise ValueError(f"Token ID out of range [0, {vocab_size})")
    out = w[flat_ids]
    id_shape = tuple(int(x) for x in np.asarray(ids).shape)
    if id_shape:
        return out.reshape(*id_shape, embed_dim)
    return out.reshape(-1, embed_dim)


def embedding_scatter_add(grad_weights, ids, grad_out):
    """Accumulate embedding gradients with ``add.at`` (backward pass).

    Args:
        grad_weights (array-like): Mutable weight gradient buffer of shape
            ``(vocab_size, embed_dim)``. Updated in place.
        ids (array-like): int64 token IDs used in the forward gather, shape ``(n,)``.
        grad_out (array-like): Upstream gradient rows, shape ``(n, embed_dim)``.

    Returns:
        ndarray: The same ``grad_weights`` buffer after accumulation.

    Raises:
        ValueError: If the token IDs are out of range [0, vocab_size)
            or if the lengths of the token IDs and gradient output do not match.
        RuntimeError: If the token IDs are not on the same device as the gradient weights.
    """
    grad_w = _on_device(grad_weights, dtype=np_backend.dtype)
    vocab_size, embed_dim = grad_w.shape

    ids_arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    grad_flat = _on_device(grad_out, dtype=np_backend.dtype).reshape(-1, embed_dim)

    if ids_arr.size != grad_flat.shape[0]:
        raise ValueError(
            f"ids length ({ids_arr.size}) must match grad_out rows ({grad_flat.shape[0]})."
        )
    if ids_arr.size and (int(ids_arr.min()) < 0 or int(ids_arr.max()) >= vocab_size):
        raise ValueError(f"Token ID out of range [0, {vocab_size})")

    if ids_arr.__class__.__module__.startswith("cupy") or hasattr(grad_w, "device"):
        ids_dev = _on_device(ids_arr, dtype=np.int64)
    else:
        ids_dev = host_np.asarray(ids_arr, dtype=host_np.int64).reshape(-1)
        if not ids_dev.flags.writeable:
            ids_dev = ids_dev.copy()

    np.add.at(grad_w, ids_dev, grad_flat)
    return grad_w

"""Embedding gather (forward) and scatter-add (backward)."""
from __future__ import annotations

import os

from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, on_device, using_gpu

# Bounds checks force a GPU sync (min/max). Off by default for training speed;
# set NIMBLEML_CHECK_BOUNDS=1 to enable.
_CHECK_BOUNDS = os.environ.get("NIMBLEML_CHECK_BOUNDS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _maybe_check_ids(flat_ids, vocab_size: int) -> None:
    if not _CHECK_BOUNDS or flat_ids.size == 0:
        return
    lo = int(flat_ids.min())
    hi = int(flat_ids.max())
    if lo < 0 or hi >= vocab_size:
        raise ValueError(f"Token ID out of range [0, {vocab_size})")


def embedding_lookup(weights, ids):
    """Gather embedding rows (forward pass)."""
    w = on_device(weights, dtype=np_backend.dtype)
    vocab_size, embed_dim = w.shape
    ids_arr = on_device(ids, dtype=np.int64)
    flat_ids = ids_arr.reshape(-1)
    _maybe_check_ids(flat_ids, vocab_size)
    out = w[flat_ids]
    id_shape = ids_arr.shape
    if id_shape:
        return out.reshape(*id_shape, embed_dim)
    return out.reshape(-1, embed_dim)


def embedding_scatter_add(grad_weights, ids, grad_out):
    """Accumulate embedding gradients into ``grad_weights`` in place.

    ``grad_weights`` is used as-is in its own dtype — under fp16 compute the
    parameter grad buffer is fp32, and casting it here would scatter into a
    throwaway copy (silently dropping every embedding gradient).
    """
    grad_w = grad_weights
    if not isinstance(grad_w, np.ndarray):
        # Foreign (e.g. host NumPy under CuPy backend): move to the backend,
        # keeping the buffer dtype. Callers use the returned array.
        grad_w = np.asarray(grad_w)
    vocab_size, embed_dim = grad_w.shape
    ids_arr = on_device(ids, dtype=np.int64).reshape(-1)
    grad_flat = on_device(grad_out, dtype=np_backend.dtype).reshape(-1, embed_dim)
    if ids_arr.size != grad_flat.shape[0]:
        raise ValueError(
            f"ids length ({ids_arr.size}) must match grad_out rows ({grad_flat.shape[0]})."
        )
    _maybe_check_ids(ids_arr, vocab_size)

    if grad_flat.dtype != grad_w.dtype:
        grad_flat = grad_flat.astype(grad_w.dtype)

    if using_gpu:
        # CuPy: ``add.at`` handles duplicate indices without host roundtrip.
        np.add.at(grad_w, ids_arr, grad_flat)
        return grad_w

    if grad_w.dtype == np.float32:
        from NimbleML._native_loader import native
        import numpy as host_np

        gw = host_np.ascontiguousarray(host_np.asarray(grad_w, dtype=host_np.float32))
        ids_host = host_np.ascontiguousarray(host_np.asarray(ids_arr, dtype=host_np.int64))
        go = host_np.ascontiguousarray(host_np.asarray(grad_flat, dtype=host_np.float32))
        native.embedding_scatter_add(gw, ids_host, go)
        if gw is not grad_w:
            grad_w[...] = gw
        return grad_w

    np.add.at(grad_w, ids_arr, grad_flat)
    return grad_w

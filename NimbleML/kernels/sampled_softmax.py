"""Sampled softmax cross-entropy on a label + negative class subset.

Trains with softmax over ``num_samples`` negatives plus the true class instead
of the full vocabulary. Gradients scatter back into the full logits tensor.
"""
from __future__ import annotations
import numpy as host_np
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, as_label_indices, on_device


def sample_negative_indices(vocab_size, label_indices, num_samples, *, rng=None):
    """Sample ``num_samples`` negative class IDs per row (never equal to the label).

    Args:
        vocab_size (int): Number of classes.
        label_indices (array-like): True labels ``(batch,)``.
        num_samples (int): Negatives per row.
        rng: Optional ``numpy.random.Generator``; defaults to NumPy global RNG.

    Returns:
        ndarray: int64 array of shape ``(batch, num_samples)``.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2.")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if num_samples >= vocab_size:
        raise ValueError("num_samples must be smaller than vocab_size.")

    labels = host_np.asarray(label_indices, dtype=host_np.int64).reshape(-1)
    batch = labels.size
    generator = rng if hasattr(rng, "integers") else host_np.random.default_rng()
    all_ids = host_np.arange(vocab_size, dtype=host_np.int64)
    negatives = host_np.empty((batch, num_samples), dtype=host_np.int64)

    for row in range(batch):
        choices = host_np.delete(all_ids, int(labels[row]))
        negatives[row] = generator.choice(choices, size=num_samples, replace=False)

    return negatives


def _candidate_ids(label_indices, negative_indices):
    labels = np.asarray(label_indices, dtype=np.int64).reshape(-1, 1)
    negs = np.asarray(negative_indices, dtype=np.int64)
    if negs.ndim != 2 or negs.shape[0] != labels.shape[0]:
        raise ValueError("negative_indices must have shape (batch, num_samples).")
    return np.concatenate([labels, negs], axis=1)


def _gather_logits(logits, candidate_ids):
    batch = candidate_ids.shape[0]
    rows = np.arange(batch, dtype=np.int64)[:, None]
    return logits[rows, candidate_ids]


def _scatter_logits_grad(grad_logits, candidate_ids, grad_sampled):
    batch, width = candidate_ids.shape
    flat_rows = np.repeat(np.arange(batch, dtype=np.int64), width)
    flat_cols = candidate_ids.reshape(-1)
    np.add.at(grad_logits, (flat_rows, flat_cols), grad_sampled.reshape(-1))


def sampled_softmax_forward(logits, label_indices, negative_indices):
    """Mean sampled cross-entropy; true class is always candidate column 0.

    Returns:
        tuple: ``(loss, logits, candidate_ids, max_vals, sum_exp)``.
    """
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size, vocab_size = logits_arr.shape
    labels = as_label_indices(label_indices, batch_size=batch_size)
    candidate_ids = _candidate_ids(labels, negative_indices)
    sampled_logits = _gather_logits(logits_arr, candidate_ids)

    zero_targets = np.zeros(batch_size, dtype=np.int64)
    loss, _, max_vals, sum_exp = fused_crossentropy_forward(sampled_logits, zero_targets)
    return loss, logits_arr, candidate_ids, max_vals, sum_exp


def sampled_softmax_backward(grad_scale, logits, candidate_ids, max_vals, sum_exp):
    """Gradient w.r.t. full ``logits`` from sampled softmax CE."""
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size = candidate_ids.shape[0]
    sampled_logits = _gather_logits(logits_arr, candidate_ids)
    zero_targets = np.zeros(batch_size, dtype=np.int64)
    grad_sampled = fused_crossentropy_backward(
        grad_scale,
        sampled_logits,
        zero_targets,
        max_vals,
        sum_exp,
    )
    grad_logits = np.zeros_like(logits_arr)
    _scatter_logits_grad(grad_logits, candidate_ids, grad_sampled)
    return grad_logits

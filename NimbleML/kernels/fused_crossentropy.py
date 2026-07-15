"""Fused log-softmax + cross-entropy on the active NumPy/CuPy backend.

For fp16 inputs, reductions (``sum_exp``, per-sample loss sum) run with fp32
accumulators: an fp16 loss sum overflows at 65504 (already at ~6k rows of
loss ~10), and fp16 ``sum_exp`` loses several bits per row. fp32/fp64 inputs
keep single-dtype math so finite-difference gradchecks stay exact.
"""
from __future__ import annotations
from NimbleML._native_loader import native as _native  # noqa: F401  # required
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_label_indices, np, on_device


def fused_crossentropy_forward(logits, label_indices):
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size, _ = logits_arr.shape
    labels = as_label_indices(label_indices, batch_size=batch_size)

    acc = np.float32 if logits_arr.dtype == np.float16 else logits_arr.dtype

    max_vals = np.max(logits_arr, axis=1, keepdims=True)
    shifted = logits_arr - max_vals
    sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True, dtype=acc)
    log_sum_exp = max_vals.ravel().astype(acc, copy=False) + np.log(sum_exp.ravel())

    row_idx = np.arange(batch_size, dtype=np.int64)
    correct_logits = logits_arr[row_idx, labels]
    per_sample = log_sum_exp - correct_logits
    loss = float(np.sum(per_sample) / batch_size)
    return loss, logits_arr, max_vals, sum_exp


def fused_crossentropy_backward(grad_scale, logits, label_indices, max_vals, sum_exp):
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size, _ = logits_arr.shape
    labels = as_label_indices(label_indices, batch_size=batch_size)
    max_arr = on_device(max_vals, dtype=logits_arr.dtype)
    # sum_exp keeps its (possibly wider) accumulator dtype from the forward.
    sum_arr = np.asarray(sum_exp)

    shifted = logits_arr - max_arr
    probs = np.exp(shifted)
    # Fold 1/sum into the logits dtype per row so the big (batch, vocab) buffer
    # stays in compute precision (keeps fp16 GEMMs downstream).
    inv_sum = (1.0 / sum_arr).astype(logits_arr.dtype, copy=False)
    probs *= inv_sum
    row_idx = np.arange(batch_size, dtype=np.int64)
    probs[row_idx, labels] -= 1.0
    probs *= logits_arr.dtype.type(float(grad_scale) / batch_size)
    return probs

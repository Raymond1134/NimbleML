"""Per-sequence summed NLL through the tied LM head (DPO, sequence scoring).

``compute_loss`` averages over every supervised token in the batch, which
mixes sequences together; preference losses need one log-prob per row.
"""
from __future__ import annotations

import numpy as host_np

from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_label_indices, np
from NimbleML.utils.tensor import Tensor

_MAX_LOGIT_BYTES = 1024 ** 3


def _to_host(arr):
    return arr.get() if hasattr(arr, "get") else host_np.asarray(arr)


def tied_sequence_nll(hidden, embedding_weights, labels, ignore_index: int) -> Tensor:
    """Summed NLL of each batch row, shape ``(batch,)``.

    Tensor data is in the compute dtype; ``out.row_nll`` (float64) and
    ``out.row_tokens`` (supervised token counts) are exact host copies for
    loss bookkeeping. Backward takes one gradient per row.
    """
    if hidden.ndim != 3:
        raise ValueError("tied_sequence_nll expects hidden of shape (batch, seq, d_model).")
    batch, seq, d_model = hidden.shape
    rows = batch * seq
    h_all = hidden._view((rows, d_model))
    w = embedding_weights._view()
    lab = as_label_indices(labels, batch_size=rows)
    idx = np.nonzero(lab != ignore_index)[0]
    n = int(idx.size)
    row_of = idx // seq
    h = h_all[idx]
    y = lab[idx]
    vocab = int(w.shape[0])
    w_T = np.swapaxes(w, -2, -1)
    chunk = max(1, min(max(n, 1), _MAX_LOGIT_BYTES // (vocab * int(np.dtype(h.dtype).itemsize))))

    tok_nll = np.zeros((n,), dtype=np.float32)
    stats = []
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        logits = np.matmul(h[sl], w_T)
        _, logits_arr, m_raw, s_raw = fused_crossentropy_forward(logits, y[sl])
        m = np.asarray(m_raw).reshape(-1).astype(np.float32, copy=False)
        s = np.asarray(s_raw).reshape(-1).astype(np.float32, copy=False)
        correct = logits_arr[np.arange(sl.stop - sl.start), y[sl]].astype(np.float32)
        tok_nll[sl] = m + np.log(s) - correct
        stats.append((sl, m_raw, s_raw))
        del logits, logits_arr

    row_nll = np.bincount(row_of, weights=tok_nll, minlength=batch) if n else np.zeros((batch,))
    row_tok = np.bincount(row_of, minlength=batch) if n else np.zeros((batch,), dtype=np.int64)
    out = Tensor(
        np.asarray(row_nll, dtype=np_backend.dtype).reshape(-1),
        (batch,),
        requires_grad=hidden.requires_grad or embedding_weights.requires_grad,
        _children=(hidden, embedding_weights),
        _op="tied_sequence_nll",
    )
    out.row_nll = _to_host(row_nll).astype(host_np.float64)
    out.row_tokens = _to_host(row_tok).astype(host_np.int64)

    def _backward():
        if out.grad is None or n == 0:
            return
        g = np.asarray(out.grad, dtype=np.float32).reshape(-1)
        tok_scale = g[row_of]
        grad_h = np.empty_like(h)
        grad_w = np.zeros_like(w) if embedding_weights.requires_grad else None
        for sl, m, s in stats:
            logits = np.matmul(h[sl], w_T)
            # grad_scale == rows makes the kernel emit plain (softmax - onehot).
            gl = fused_crossentropy_backward(float(sl.stop - sl.start), logits, y[sl], m, s)
            gl *= tok_scale[sl].astype(gl.dtype)[:, None]
            grad_h[sl] = np.matmul(gl, w)
            if grad_w is not None:
                np.add(grad_w, np.matmul(gl.T, h[sl]), out=grad_w)
            del logits, gl
        if hidden.requires_grad:
            full = np.zeros((rows, d_model), dtype=np_backend.dtype)
            full[idx] = grad_h
            hidden._accumulate_grad(full.ravel())
        if grad_w is not None:
            embedding_weights._accumulate_grad(grad_w.ravel())

    out._backward = _backward
    return out


def weighted_sum(values: Tensor, weights) -> Tensor:
    """Scalar ``sum_i weights[i] * values[i]`` with constant (host) weights."""
    wts = np.asarray(host_np.asarray(weights, dtype=host_np.float32).reshape(-1))
    total = float((values.data.astype(np.float32) * wts).sum())
    out = Tensor(
        [total],
        (),
        requires_grad=values.requires_grad,
        _children=(values,),
        _op="weighted_sum",
    )

    def _backward():
        if out.grad is None:
            return
        g = float(out.grad[0])
        values._accumulate_grad((wts * g).astype(np_backend.dtype))

    out._backward = _backward
    return out

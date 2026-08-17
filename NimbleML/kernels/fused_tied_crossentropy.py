"""Fused tied LM head + cross-entropy (hidden @ W^T then CE)."""
from __future__ import annotations
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_label_indices, np, on_device

_MAX_LOGIT_BYTES = 3 * 1024 ** 3


def _row_chunk(rows: int, vocab: int, itemsize: int) -> int:
    nbytes = rows * vocab * max(1, itemsize)
    if nbytes <= _MAX_LOGIT_BYTES:
        return rows
    per_row = vocab * max(1, itemsize)
    return max(1, min(rows, _MAX_LOGIT_BYTES // per_row))


def fused_tied_crossentropy_forward(hidden, weights, label_indices):
    """Mean CE from hidden states and tied embedding weights ``(vocab, d_model)``.

    When ``rows × vocab`` logits would exceed ~3 GiB, the vocab GEMM is tiled.
    Backward recomputes the same tiles (identical math, same ``token_emb.weights``).
    """
    h = on_device(hidden, dtype=np_backend.dtype)
    w = on_device(weights, dtype=np_backend.dtype)
    rows = int(h.shape[0])
    vocab = int(w.shape[0])
    labels = as_label_indices(label_indices, batch_size=rows)
    chunk = _row_chunk(rows, vocab, int(np.dtype(h.dtype).itemsize))
    w_T = np.swapaxes(w, -2, -1)
    if chunk >= rows:
        logits = np.matmul(h, w_T)
        loss, logits_arr, max_vals, sum_exp = fused_crossentropy_forward(logits, labels)
        return loss, h, w, logits_arr, max_vals, sum_exp

    loss_acc = None
    for start in range(0, rows, chunk):
        sl = slice(start, min(start + chunk, rows))
        n = sl.stop - sl.start
        logits = np.matmul(h[sl], w_T)
        loss_c, _, _, _ = fused_crossentropy_forward(logits, labels[sl])
        term = loss_c * n
        loss_acc = term if loss_acc is None else loss_acc + term
        del logits
    return loss_acc / rows, h, w, None, None, None


def fused_tied_crossentropy_backward(grad_scale, hidden, weights, label_indices, logits, max_vals, sum_exp):
    h = on_device(hidden, dtype=np_backend.dtype)
    w = on_device(weights, dtype=np_backend.dtype)
    rows = int(h.shape[0])
    labels = as_label_indices(label_indices, batch_size=rows)
    if logits is not None:
        grad_logits = fused_crossentropy_backward(grad_scale, logits, labels, max_vals, sum_exp)
        grad_h = np.matmul(grad_logits, w)
        grad_w = np.matmul(grad_logits.T, h)
        return grad_h, grad_w

    vocab = int(w.shape[0])
    chunk = _row_chunk(rows, vocab, int(np.dtype(h.dtype).itemsize))
    w_T = np.swapaxes(w, -2, -1)
    grad_h = np.empty_like(h)
    grad_w = np.zeros_like(w)
    for start in range(0, rows, chunk):
        sl = slice(start, min(start + chunk, rows))
        n = sl.stop - sl.start
        logits_c = np.matmul(h[sl], w_T)
        _, logits_arr, max_c, sum_c = fused_crossentropy_forward(logits_c, labels[sl])
        scale = float(grad_scale) * (n / float(rows))
        grad_logits = fused_crossentropy_backward(scale, logits_arr, labels[sl], max_c, sum_c)
        grad_h[sl] = np.matmul(grad_logits, w)
        np.add(grad_w, np.matmul(grad_logits.T, h[sl]), out=grad_w)
        del logits_c, logits_arr, grad_logits
    return grad_h, grad_w

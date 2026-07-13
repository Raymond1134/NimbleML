"""Fused scaled dot-product attention on the active NumPy/CuPy backend.

Modes (``NIMBLEML_SDPA``):
  - ``torch`` / ``auto`` (when PyTorch+CUDA installed): ``F.scaled_dot_product_attention``
    via DLPack — same FlashAttention / mem-efficient kernels as PyTorch training.
  - ``matmul`` / ``dense``: CuPy/cuBLAS materializing ``(S,S)`` scores.
  - ``flash`` / ``tiled``: in-repo CUDA FA (memory opt-in; slower than torch SDPA).
"""
from __future__ import annotations

import os

from NimbleML._native_loader import native as _native
from NimbleML.utils.axis import normalize_axis
from NimbleML.utils.np_backend import np, using_gpu

_SDPA_MODE = os.environ.get("NIMBLEML_SDPA", "auto").strip().lower()
_TILE = int(os.environ.get("NIMBLEML_SDPA_TILE", "128"))


def _softmax_forward(arr, axis: int = -1):
    axis = normalize_axis(arr.ndim, axis)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    exps = np.exp(arr - max_vals)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def _softmax_backward(grad_out, probs, axis: int = -1):
    axis = normalize_axis(probs.ndim, axis)
    dot = np.sum(grad_out * probs, axis=axis, keepdims=True)
    return probs * (grad_out - dot)


def _native_fa_available() -> bool:
    try:
        return bool(getattr(_native, "flash_sdpa_available", lambda: False)()) and hasattr(
            _native, "flash_sdpa_forward_device"
        )
    except Exception:
        return False


def _use_flash(seq: int, *, causal: bool) -> bool:
    """When to use FlashAttention instead of dense matmul.

    Device FA (``NIMBLEML_SDPA=flash``) saves the ``S×S`` score buffer but the
    current CUDA kernel is a simple per-row loop — much slower than cuBLAS at
    typical GPT lengths (≤4k). ``auto`` therefore keeps dense matmul for speed;
    set ``flash`` when VRAM-bound.
    """
    del seq  # reserved for future auto heuristics (e.g. extreme seq)
    if not causal:
        return False
    if _SDPA_MODE in ("flash", "tiled"):
        return bool(using_gpu)
    # auto / matmul / dense → cuBLAS path
    return False


def _ptr(arr) -> int:
    """Device data pointer for a contiguous CuPy array."""
    return int(arr.data.ptr)


def _as_f32_contig(arr):
    a = np.ascontiguousarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
        a = np.ascontiguousarray(a)
    return a


def _matmul_sdpa_forward(q, k, v, scale, mask_arr=None):
    scores = np.matmul(q, np.swapaxes(k, -2, -1))
    np.divide(scores, scale, out=scores)
    if mask_arr is not None:
        np.add(scores, mask_arr, out=scores)
    probs = _softmax_forward(scores, axis=-1)
    out = np.matmul(probs, v)
    return out, probs


def _matmul_sdpa_backward(grad_out, q, k, v, probs, scale):
    grad_probs = np.matmul(grad_out, np.swapaxes(v, -2, -1))
    grad_scores = _softmax_backward(grad_probs, probs, axis=-1)
    np.divide(grad_scores, scale, out=grad_scores)
    grad_q = np.matmul(grad_scores, k)
    grad_k = np.matmul(np.swapaxes(grad_scores, -2, -1), q)
    grad_v = np.matmul(np.swapaxes(probs, -2, -1), grad_out)
    return grad_q, grad_k, grad_v


def _native_flash_forward(q, k, v, scale):
    """On-device FA; ``scale`` is sqrt(dk) (we pass 1/scale to CUDA)."""
    bh, seq, dk = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    qf = _as_f32_contig(q)
    kf = _as_f32_contig(k)
    vf = _as_f32_contig(v)
    out = np.empty((bh, seq, dk), dtype=np.float32)
    m = np.empty((bh, seq), dtype=np.float32)
    l = np.empty((bh, seq), dtype=np.float32)
    inv_scale = float(1.0 / scale)
    _native.flash_sdpa_forward_device(
        _ptr(qf), _ptr(kf), _ptr(vf), _ptr(out), _ptr(m), _ptr(l),
        bh, seq, dk, inv_scale,
    )
    return out.astype(q.dtype, copy=False), {
        "m": m,
        "l": l,
        "flash": True,
        "native": True,
        "scale": float(scale),
        "q_f32": qf,
        "k_f32": kf,
        "v_f32": vf,
    }


def _native_flash_backward(grad_out, q, k, v, meta):
    bh, seq, dk = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    qf = meta.get("q_f32")
    kf = meta.get("k_f32")
    vf = meta.get("v_f32")
    if qf is None:
        qf = _as_f32_contig(q)
        kf = _as_f32_contig(k)
        vf = _as_f32_contig(v)
    go = _as_f32_contig(grad_out)
    m = np.ascontiguousarray(meta["m"], dtype=np.float32)
    l = np.ascontiguousarray(meta["l"], dtype=np.float32)
    gq = np.empty_like(qf)
    gk = np.empty_like(kf)
    gv = np.empty_like(vf)
    inv_scale = float(1.0 / float(meta["scale"]))
    _native.flash_sdpa_backward_device(
        _ptr(go), _ptr(qf), _ptr(kf), _ptr(vf), _ptr(m), _ptr(l),
        _ptr(gq), _ptr(gk), _ptr(gv), bh, seq, dk, inv_scale,
    )
    return (
        gq.astype(q.dtype, copy=False),
        gk.astype(k.dtype, copy=False),
        gv.astype(v.dtype, copy=False),
    )


def _flash_causal_forward(q, k, v, scale, tile: int):
    """Python tiled online-softmax (fallback when native FA is unavailable)."""
    bh, seq, dk = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    qf = q.astype(np.float32, copy=False)
    kf = k.astype(np.float32, copy=False)
    vf = v.astype(np.float32, copy=False)
    out = np.zeros((bh, seq, dk), dtype=np.float32)
    m_i = np.full((bh, seq), np.float32(-1e9), dtype=np.float32)
    l_i = np.zeros((bh, seq), dtype=np.float32)

    for i0 in range(0, seq, tile):
        i1 = min(seq, i0 + tile)
        q_blk = qf[:, i0:i1, :]
        o_blk = np.zeros((bh, i1 - i0, dk), dtype=np.float32)
        m_blk = np.full((bh, i1 - i0), np.float32(-1e9), dtype=np.float32)
        l_blk = np.zeros((bh, i1 - i0), dtype=np.float32)

        for j0 in range(0, i1, tile):
            j1 = min(seq, j0 + tile)
            k_blk = kf[:, j0:j1, :]
            v_blk = vf[:, j0:j1, :]
            scores = np.matmul(q_blk, np.swapaxes(k_blk, -2, -1))
            scores *= np.float32(1.0 / scale)

            qi = np.arange(i0, i1, dtype=np.int32).reshape(1, -1, 1)
            kj = np.arange(j0, j1, dtype=np.int32).reshape(1, 1, -1)
            if using_gpu:
                qi = np.asarray(qi)
                kj = np.asarray(kj)
            causal = kj <= qi
            neg = np.float32(-1e9)
            scores = np.where(causal, scores, neg)

            block_max = np.max(scores, axis=-1)
            m_new = np.maximum(m_blk, block_max)
            alpha = np.exp(m_blk - m_new)
            p = np.exp(scores - m_new[:, :, None])
            p = np.where(causal, p, np.float32(0.0))
            l_new = alpha * l_blk + np.sum(p, axis=-1)
            o_blk = o_blk * alpha[:, :, None] + np.matmul(p, v_blk)
            m_blk = m_new
            l_blk = l_new

        inv = np.reciprocal(np.maximum(l_blk, np.float32(1e-12)))
        out[:, i0:i1, :] = o_blk * inv[:, :, None]
        m_i[:, i0:i1] = m_blk
        l_i[:, i0:i1] = l_blk

    return out.astype(q.dtype, copy=False), {
        "m": m_i,
        "l": l_i,
        "flash": True,
        "native": False,
        "scale": float(scale),
    }


def _flash_causal_backward(grad_out, q, k, v, meta):
    if meta.get("native"):
        return _native_flash_backward(grad_out, q, k, v, meta)

    scale = float(meta["scale"])
    bh, seq, dk = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    tile = _TILE
    qf = q.astype(np.float32, copy=False)
    kf = k.astype(np.float32, copy=False)
    vf = v.astype(np.float32, copy=False)
    go = grad_out.astype(np.float32, copy=False)

    grad_q = np.zeros_like(qf)
    grad_k = np.zeros_like(kf)
    grad_v = np.zeros_like(vf)

    for i0 in range(0, seq, tile):
        i1 = min(seq, i0 + tile)
        q_blk = qf[:, i0:i1, :]
        go_blk = go[:, i0:i1, :]
        for j0 in range(0, i1, tile):
            j1 = min(seq, j0 + tile)
            k_blk = kf[:, j0:j1, :]
            v_blk = vf[:, j0:j1, :]

            scores = np.matmul(q_blk, np.swapaxes(k_blk, -2, -1))
            scores *= np.float32(1.0 / scale)
            qi = np.arange(i0, i1, dtype=np.int32).reshape(1, -1, 1)
            kj = np.arange(j0, j1, dtype=np.int32).reshape(1, 1, -1)
            if using_gpu:
                qi = np.asarray(qi)
                kj = np.asarray(kj)
            causal = kj <= qi
            scores = np.where(causal, scores, np.float32(-1e9))
            probs = _softmax_forward(scores, axis=-1)
            probs = np.where(causal, probs, np.float32(0.0))

            grad_v[:, j0:j1, :] += np.matmul(np.swapaxes(probs, -2, -1), go_blk)
            dP = np.matmul(go_blk, np.swapaxes(v_blk, -2, -1))
            dS = _softmax_backward(dP, probs, axis=-1)
            dS = np.where(causal, dS, np.float32(0.0))
            dS *= np.float32(1.0 / scale)
            grad_q[:, i0:i1, :] += np.matmul(dS, k_blk)
            grad_k[:, j0:j1, :] += np.matmul(np.swapaxes(dS, -2, -1), q_blk)

    return (
        grad_q.astype(q.dtype, copy=False),
        grad_k.astype(k.dtype, copy=False),
        grad_v.astype(v.dtype, copy=False),
    )


_TORCH_FALLBACK_WARNED = False


def fused_sdpa_forward(q, k, v, scale, mask_arr=None, *, batch=None, num_heads=None):
    """Returns ``(out, ctx)`` where ``ctx`` is probs ndarray or flash/torch meta dict."""
    seq = int(q.shape[-2])
    causal = mask_arr is not None

    # Prefer PyTorch fused SDPA (FlashAttention / mem-efficient) when available.
    if causal and using_gpu:
        from NimbleML.kernels.torch_sdpa import _want_torch_sdpa, torch_sdpa_forward

        if _want_torch_sdpa():
            try:
                return torch_sdpa_forward(
                    q, k, v, scale, causal=True, batch=batch, num_heads=num_heads
                )
            except Exception as exc:
                # Strict mode: surface the error so we never silently run dense S×S.
                if _SDPA_MODE in ("torch", "pytorch", "sdpa"):
                    raise
                global _TORCH_FALLBACK_WARNED
                if not _TORCH_FALLBACK_WARNED:
                    _TORCH_FALLBACK_WARNED = True
                    import warnings

                    warnings.warn(
                        f"torch SDPA failed ({exc}); falling back to CuPy path",
                        stacklevel=2,
                    )

    if _use_flash(seq, causal=causal):
        if _native_fa_available():
            return _native_flash_forward(q, k, v, scale)
        return _flash_causal_forward(q, k, v, scale, _TILE)
    return _matmul_sdpa_forward(q, k, v, scale, mask_arr)


def fused_sdpa_backward(grad_out, q, k, v, probs_or_meta, scale):
    if isinstance(probs_or_meta, dict) and probs_or_meta.get("torch_sdpa"):
        from NimbleML.kernels.torch_sdpa import torch_sdpa_backward

        return torch_sdpa_backward(grad_out, probs_or_meta)
    if isinstance(probs_or_meta, dict) and probs_or_meta.get("flash"):
        return _flash_causal_backward(grad_out, q, k, v, probs_or_meta)
    return _matmul_sdpa_backward(grad_out, q, k, v, probs_or_meta, scale)

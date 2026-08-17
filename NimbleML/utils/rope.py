"""Rotary positional embeddings (RoPE), GPT-NeoX half-split convention.

The cache is ``(seq_len, head_dim)`` with the frequency table duplicated across
both halves, so ``cos[:, :half] == cos[:, half:]``. ``apply_rope_flat`` rotates
dimension pair ``(i, i + half)`` by ``theta_i * position``.
"""
from __future__ import annotations
from NimbleML.utils.np_backend import np, using_gpu

_rope_kernel = None


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / float(half)))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _gpu_rope_kernel():
    global _rope_kernel
    if _rope_kernel is None:
        import cupy as cp

        _rope_kernel = cp.ElementwiseKernel(
            "T x1, T x2, T c, T s",
            "T y1, T y2",
            "y1 = x1 * c - x2 * s; y2 = x1 * s + x2 * c;",
            "nimbleml_rope_pair",
        )
    return _rope_kernel


def apply_rope_flat(x, cos, sin, *, inverse: bool = False):
    seq = x.shape[-2]
    half = x.shape[-1] // 2
    c = cos[:seq, :half].astype(x.dtype, copy=False)
    s = sin[:seq, :half].astype(x.dtype, copy=False)
    if inverse:
        s = -s
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    if using_gpu:
        try:
            _gpu_rope_kernel()(x1, x2, c, s, out[..., :half], out[..., half:])
            return out
        except Exception:
            pass
    out[..., :half] = x1 * c - x2 * s
    out[..., half:] = x1 * s + x2 * c
    return out


def apply_rope(x, cos, sin):
    return apply_rope_flat(x, cos, sin)

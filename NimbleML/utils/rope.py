"""Rotary positional embeddings (RoPE), GPT-NeoX half-split convention.

The cache is ``(seq_len, head_dim)`` with the frequency table duplicated across
both halves, so ``cos[:, :half] == cos[:, half:]``. ``apply_rope_flat`` rotates
dimension pair ``(i, i + half)`` by ``theta_i * position`` — an orthogonal
rotation, so the inverse (used by backward) is the same map with ``-sin``.
"""
from __future__ import annotations
from NimbleML.utils.np_backend import np


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    """Return ``(cos, sin)`` caches of shape ``(seq_len, head_dim)`` in float32."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / float(half)))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope_flat(x, cos, sin, *, inverse: bool = False):
    """Apply RoPE to ``x`` of shape ``(..., seq, head_dim)``.

    Args:
        x: Query/key activations; the last two axes must be ``(seq, head_dim)``.
        cos, sin: Caches from :func:`build_rope_cache` (``head_dim`` matching ``x``).
        inverse: Apply the transpose rotation (exact backward of the forward map).

    Returns:
        ndarray: Rotated array, same shape and dtype as ``x``.
    """
    seq = x.shape[-2]
    half = x.shape[-1] // 2
    c = cos[:seq, :half].astype(x.dtype, copy=False)
    s = sin[:seq, :half].astype(x.dtype, copy=False)
    if inverse:
        s = -s
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * c - x2 * s
    out[..., half:] = x1 * s + x2 * c
    return out


def apply_rope(x, cos, sin):
    """Apply RoPE to ``x`` of shape ``(batch, heads, seq, head_dim)``."""
    return apply_rope_flat(x, cos, sin)

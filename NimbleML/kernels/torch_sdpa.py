"""Optional PyTorch SDPA backend for NimbleML attention (CuPy <-> DLPack).

When ``torch`` + CUDA is available and ``NIMBLEML_SDPA=torch`` (or ``auto``
with torch present), causal attention uses ``F.scaled_dot_product_attention``
— the same FlashAttention / mem-efficient kernels PyTorch training uses.

Critical: Q/K/V must be **device-contiguous** for zero-copy DLPack. A host
fallback here destroys throughput (PCIe round-trips every layer).
"""
from __future__ import annotations

import os
import warnings

_TORCH_OK = None
_HOST_FALLBACK_WARNED = False
_STATS = {"forward_calls": 0, "host_fallback": 0, "backend": None}


def torch_sdpa_available() -> bool:
    global _TORCH_OK
    if _TORCH_OK is not None:
        return _TORCH_OK
    try:
        import torch

        _TORCH_OK = bool(torch.cuda.is_available())
    except Exception:
        _TORCH_OK = False
    return _TORCH_OK


def _want_torch_sdpa() -> bool:
    mode = os.environ.get("NIMBLEML_SDPA", "auto").strip().lower()
    if mode in ("torch", "pytorch", "sdpa"):
        return torch_sdpa_available()
    if mode in ("auto", ""):
        # Prefer torch SDPA whenever it is installed — matches PyTorch league.
        return torch_sdpa_available()
    return False


def torch_sdpa_stats() -> dict:
    """Diagnostic counters (calls, host fallbacks) for benches / logs."""
    return dict(_STATS)


def _as_contiguous_cupy(arr):
    import cupy as cp

    if not isinstance(arr, cp.ndarray):
        arr = cp.asarray(arr)
    if not arr.flags.c_contiguous:
        arr = cp.ascontiguousarray(arr)
    return arr


def _cupy_to_torch(arr, *, allow_host: bool = False):
    """Zero-copy CuPy → Torch via DLPack. Contiguous device memory required."""
    import torch

    arr = _as_contiguous_cupy(arr)
    try:
        if hasattr(arr, "__dlpack__"):
            return torch.from_dlpack(arr)
        return torch.utils.dlpack.from_dlpack(arr.toDlpack())
    except Exception as exc:
        _STATS["host_fallback"] += 1
        if not allow_host:
            raise RuntimeError(
                "CuPy→Torch DLPack failed (refusing host fallback; "
                "check contiguous GPU tensors / matching CUDA runtimes)"
            ) from exc
        global _HOST_FALLBACK_WARNED
        if not _HOST_FALLBACK_WARNED:
            _HOST_FALLBACK_WARNED = True
            warnings.warn(
                f"CuPy→Torch DLPack failed ({exc}); using host round-trip "
                "(severe throughput hit). Fix contiguous / CUDA interop.",
                stacklevel=2,
            )
        import numpy as host_np

        host = host_np.asarray(arr.get())
        return torch.as_tensor(host, device="cuda")


def _torch_to_cupy(t):
    import cupy as cp

    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    try:
        return cp.from_dlpack(t)
    except Exception as exc:
        raise RuntimeError(
            "Torch→CuPy DLPack failed (refusing host fallback)"
        ) from exc


def _cupy_then_torch():
    """Make Torch's current stream wait for outstanding CuPy work."""
    import cupy as cp
    import torch

    if not torch.cuda.is_initialized():
        torch.cuda.init()
    try:
        t_stream = torch.cuda.current_stream()
        c_stream = cp.cuda.get_current_stream()
        ev = cp.cuda.Event()
        ev.record(c_stream)
        cp.cuda.ExternalStream(t_stream.cuda_stream).wait_event(ev)
    except Exception:
        # Last resort: stream sync (still cheaper than full device sync).
        try:
            cp.cuda.get_current_stream().synchronize()
        except Exception:
            pass


def _torch_then_cupy():
    """Make CuPy's current stream wait for outstanding Torch work."""
    import cupy as cp
    import torch

    try:
        t_stream = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(t_stream)
        # CuPy stream waits on the CUDA event underlying the torch Event.
        cp.cuda.runtime.streamWaitEvent(
            cp.cuda.get_current_stream().ptr, ev.cuda_event, 0
        )
    except Exception:
        try:
            torch.cuda.current_stream().synchronize()
        except Exception:
            pass


def torch_sdpa_forward(q, k, v, scale, *, causal: bool, batch: int | None = None, num_heads: int | None = None):
    """Run torch SDPA; returns ``(out_cupy, ctx)`` for :func:`torch_sdpa_backward`.

    Prefers 4-D ``(B, H, S, D)`` layout when ``batch`` and ``num_heads`` are given
    so FlashAttention / mem-efficient kernels dispatch the same way as PyTorch GPT.
    """
    import torch
    import torch.nn.functional as F

    _STATS["forward_calls"] += 1
    _cupy_then_torch()

    # Inputs arrive as (bh, seq, dk) from the fused MHA path.
    q = _as_contiguous_cupy(q)
    k = _as_contiguous_cupy(k)
    v = _as_contiguous_cupy(v)

    use_4d = (
        batch is not None
        and num_heads is not None
        and q.ndim == 3
        and int(q.shape[0]) == int(batch) * int(num_heads)
    )
    if use_4d:
        b, h = int(batch), int(num_heads)
        s, d = int(q.shape[1]), int(q.shape[2])
        q4 = q.reshape(b, h, s, d)
        k4 = k.reshape(b, h, s, d)
        v4 = v.reshape(b, h, s, d)
        # clone → Torch-owned leaf (FA-safe); still device-only, unlike host fallback.
        tq = _cupy_to_torch(q4).clone().detach().requires_grad_(True)
        tk = _cupy_to_torch(k4).clone().detach().requires_grad_(True)
        tv = _cupy_to_torch(v4).clone().detach().requires_grad_(True)
    else:
        tq = _cupy_to_torch(q).clone().detach().requires_grad_(True)
        tk = _cupy_to_torch(k).clone().detach().requires_grad_(True)
        tv = _cupy_to_torch(v).clone().detach().requires_grad_(True)

    # NimbleML passes scale=sqrt(dk) and divides; torch ``scale`` multiplies QK^T.
    inv_scale = 1.0 / float(scale)
    with torch.enable_grad():
        out = F.scaled_dot_product_attention(
            tq, tk, tv, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=inv_scale
        )

    _STATS["backend"] = "torch_sdpa"
    _torch_then_cupy()

    out_cp = _torch_to_cupy(out)
    if use_4d:
        # Back to (bh, seq, dk) for the fused MHA merge path.
        out_cp = out_cp.reshape(q.shape)

    if out_cp.dtype != q.dtype:
        out_cp = out_cp.astype(q.dtype, copy=False)

    ctx = {
        "torch_sdpa": True,
        "tq": tq,
        "tk": tk,
        "tv": tv,
        "out": out,
        "scale": float(scale),
        "use_4d": use_4d,
        "out_shape": q.shape,
    }
    return out_cp, ctx


def torch_sdpa_backward(grad_out, ctx):
    _cupy_then_torch()
    go = _as_contiguous_cupy(grad_out)
    if ctx.get("use_4d"):
        t = ctx["out"]
        b, h, s, d = t.shape
        go = go.reshape(b, h, s, d)
    t_go = _cupy_to_torch(go)
    if t_go.dtype != ctx["out"].dtype:
        t_go = t_go.to(dtype=ctx["out"].dtype)
    ctx["out"].backward(t_go)
    _torch_then_cupy()
    gq = _torch_to_cupy(ctx["tq"].grad)
    gk = _torch_to_cupy(ctx["tk"].grad)
    gv = _torch_to_cupy(ctx["tv"].grad)
    if ctx.get("use_4d"):
        shape = ctx["out_shape"]
        gq = gq.reshape(shape)
        gk = gk.reshape(shape)
        gv = gv.reshape(shape)
    return gq, gk, gv

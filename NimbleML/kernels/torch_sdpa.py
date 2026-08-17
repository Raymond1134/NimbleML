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


def _shared_stream() -> bool:
    from NimbleML.utils.np_backend import gpu_runtime_info

    return bool(gpu_runtime_info().get("shared_stream"))


def _sync_cupy_then_torch():
    if not _shared_stream():
        _cupy_then_torch()


def _sync_torch_then_cupy():
    if not _shared_stream():
        _torch_then_cupy()


def _torch_leaves(q, k, v, owner):
    import torch

    src_q = _cupy_to_torch(q)
    src_k = _cupy_to_torch(k)
    src_v = _cupy_to_torch(v)
    cached = None if owner is None else getattr(owner, "_sdpa_leaves", None)
    if cached is None or cached[0].shape != src_q.shape or cached[0].dtype != src_q.dtype:
        tq = src_q.clone().detach().requires_grad_(True)
        tk = src_k.clone().detach().requires_grad_(True)
        tv = src_v.clone().detach().requires_grad_(True)
        if owner is not None:
            owner._sdpa_leaves = (tq, tk, tv)
        return tq, tk, tv
    tq, tk, tv = cached
    tq.grad = None
    tk.grad = None
    tv.grad = None
    with torch.no_grad():
        tq.copy_(src_q)
        tk.copy_(src_k)
        tv.copy_(src_v)
    return tq, tk, tv


def torch_sdpa_forward(
    q, k, v, scale, *, causal: bool, batch: int | None = None, num_heads: int | None = None,
    leaves_owner=None,
):
    import torch
    import torch.nn.functional as F

    _STATS["forward_calls"] += 1
    _sync_cupy_then_torch()

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
        q_in = q.reshape(b, h, s, d)
        k_in = k.reshape(b, h, s, d)
        v_in = v.reshape(b, h, s, d)
    else:
        q_in, k_in, v_in = q, k, v

    tq, tk, tv = _torch_leaves(q_in, k_in, v_in, leaves_owner)

    inv_scale = 1.0 / float(scale)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        sdp_ctx = sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        )
    except Exception:
        from contextlib import nullcontext

        sdp_ctx = nullcontext()
    with sdp_ctx, torch.enable_grad():
        out = F.scaled_dot_product_attention(
            tq, tk, tv, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=inv_scale
        )

    _STATS["backend"] = "torch_sdpa"
    _sync_torch_then_cupy()

    out_cp = _torch_to_cupy(out)
    if use_4d:
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
    _sync_cupy_then_torch()
    go = _as_contiguous_cupy(grad_out)
    if ctx.get("use_4d"):
        t = ctx["out"]
        b, h, s, d = t.shape
        go = go.reshape(b, h, s, d)
    t_go = _cupy_to_torch(go)
    if t_go.dtype != ctx["out"].dtype:
        t_go = t_go.to(dtype=ctx["out"].dtype)
    ctx["out"].backward(t_go)
    _sync_torch_then_cupy()
    gq = _torch_to_cupy(ctx["tq"].grad)
    gk = _torch_to_cupy(ctx["tk"].grad)
    gv = _torch_to_cupy(ctx["tv"].grad)
    if ctx.get("use_4d"):
        shape = ctx["out_shape"]
        gq = gq.reshape(shape)
        gk = gk.reshape(shape)
        gv = gv.reshape(shape)
    return gq, gk, gv

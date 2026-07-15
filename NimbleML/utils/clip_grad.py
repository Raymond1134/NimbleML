"""Gradient clipping utilities."""
import math
from NimbleML.utils.np_backend import np, using_gpu


def clip_grad_norm_(params, max_norm: float) -> float:
    """Clip the total L2 norm of gradients in-place to at most ``max_norm``.

    On GPU, concatenates all grads into one buffer for a single reduction +
    scale (avoids per-parameter kernel launches / syncs that dominated the
    train-step profile vs PyTorch).

    Returns the total norm; callers should skip the optimizer step when it is
    non-finite (overflowed fp16 backward pass).
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    flats = []
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if not hasattr(grad, "ravel"):
            grad = np.asarray(grad)
            param.grad = grad
        flats.append(grad.ravel())

    if not flats:
        return 0.0

    if using_gpu and len(flats) > 1:
        return _clip_packed_gpu(flats, max_norm)

    total_sq = None
    for flat in flats:
        work = flat.astype(np.float32, copy=False) if flat.dtype == np.float16 else flat
        sq = np.dot(work, work)
        total_sq = sq if total_sq is None else total_sq + sq

    if hasattr(total_sq, "get"):
        total_sq = total_sq.get()
    total_sq = float(total_sq.item() if hasattr(total_sq, "item") else total_sq)
    total_norm = math.sqrt(total_sq)
    if total_norm == 0.0 or not math.isfinite(total_norm) or total_norm <= max_norm:
        return total_norm

    scale = max_norm / (total_norm + 1e-12)
    for flat in flats:
        np.multiply(flat, flat.dtype.type(scale), out=flat)
    return total_norm


def _clip_packed_gpu(flats, max_norm: float) -> float:
    """One concat → one L2 → optional in-place scale for all grads."""
    # Promote fp16 slices to fp32 for the norm only; scale original buffers.
    pieces = []
    for flat in flats:
        if flat.dtype == np.float16:
            pieces.append(flat.astype(np.float32, copy=False))
        else:
            pieces.append(flat)
    cat = np.concatenate(pieces)
    total_sq = np.dot(cat, cat)
    if hasattr(total_sq, "get"):
        total_sq = total_sq.get()
    total_sq = float(total_sq.item() if hasattr(total_sq, "item") else total_sq)
    total_norm = math.sqrt(total_sq)
    if total_norm == 0.0 or not math.isfinite(total_norm) or total_norm <= max_norm:
        return total_norm

    scale = max_norm / (total_norm + 1e-12)
    # Scale each original buffer (preserves dtype) — still fewer launches than
    # per-param norm reductions.
    for flat in flats:
        np.multiply(flat, flat.dtype.type(scale), out=flat)
    return total_norm

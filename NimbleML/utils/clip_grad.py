"""Gradient clipping utilities."""
import math

from NimbleML.utils.np_backend import np, dtype


def clip_grad_norm_(params, max_norm: float) -> float:
    """Clip the total L2 norm of gradients in-place to at most ``max_norm``.

    Uses one reduction over all gradients (single GPU sync on CuPy) instead of
    per-parameter ``float(np.sum(...))`` calls.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    active = []
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        active.append((param, np.asarray(grad, dtype=dtype)))

    if not active:
        return 0.0

    total_sq = None
    for _, grad in active:
        flat = grad.ravel()
        sq = np.dot(flat, flat)
        total_sq = sq if total_sq is None else total_sq + sq

    if hasattr(total_sq, "get"):
        total_sq = total_sq.get()
    if hasattr(total_sq, "item"):
        total_sq = float(total_sq.item())
    else:
        total_sq = float(total_sq)
    total_norm = math.sqrt(total_sq)
    if total_norm == 0.0 or not math.isfinite(total_norm) or total_norm <= max_norm:
        return total_norm

    scale = max_norm / total_norm
    for param, grad in active:
        np.multiply(grad, scale, out=grad)
        param.grad = grad

    return total_norm

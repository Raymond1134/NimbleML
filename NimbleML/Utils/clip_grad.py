"""Gradient clipping utilities."""
import math

from NimbleML.utils.np_backend import np, dtype


def clip_grad_norm_(params, max_norm: float) -> float:
    """Clip the total L2 norm of gradients in-place to at most ``max_norm``.

    Parameters
    ----------
    params:
        Iterable of parameters with a ``grad`` attribute (possibly ``None``).
    max_norm:
        Maximum allowed global L2 norm for all gradients.

    Returns
    -------
    float
        The original (unclipped) global L2 norm of all gradients.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    total_sq_norm = 0.0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        g = grad.get() if hasattr(grad, "get") else grad
        arr = np.asarray(g, dtype=dtype)
        total_sq_norm += float(np.sum(arr * arr))

    total_norm = math.sqrt(total_sq_norm)
    if total_norm == 0.0 or not math.isfinite(total_norm) or total_norm <= max_norm:
        return total_norm

    scale = max_norm / total_norm
    for param in params:
        if getattr(param, "grad", None) is None:
            continue
        grad = param.grad
        if hasattr(grad, "get"):
            grad = grad.get()
        grad_arr = np.asarray(grad, dtype=dtype) * scale
        param.grad = grad_arr

    return total_norm
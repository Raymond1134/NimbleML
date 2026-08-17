"""Gradient clipping utilities."""
import math
from NimbleML.utils.np_backend import np, using_gpu

_PACK_MAX = 65536


def clip_grad_norm_(params, max_norm: float, *, unscale: float = 1.0) -> float:
    """Clip the total L2 norm of gradients in-place to at most ``max_norm``.

    ``unscale`` folds GradScaler unscale into the same pass (multiply grads by
    this factor, typically ``1 / loss_scale``). Norm is computed in fp32 from
    the stored buffers; the returned value is the unscaled (true) norm.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    inv = float(unscale)
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

    total_sq = _grad_sum_squares(flats)
    if hasattr(total_sq, "get"):
        total_sq = total_sq.get()
    total_sq = float(total_sq.item() if hasattr(total_sq, "item") else total_sq)
    stored_norm = math.sqrt(total_sq)
    true_norm = stored_norm * inv
    if true_norm == 0.0 or not math.isfinite(true_norm):
        if inv != 1.0 and math.isfinite(stored_norm) and stored_norm != 0.0:
            _scale_flats(flats, inv)
        return true_norm

    clip_scale = 1.0 if true_norm <= max_norm else max_norm / (true_norm + 1e-12)
    factor = inv * clip_scale
    if factor != 1.0:
        _scale_flats(flats, factor)
    return true_norm


def _grad_sum_squares(flats):
    small = []
    total_sq = None
    for flat in flats:
        work = flat.astype(np.float32, copy=False) if flat.dtype == np.float16 else flat
        if using_gpu and work.size <= _PACK_MAX:
            small.append(work)
            continue
        sq = np.dot(work, work)
        total_sq = sq if total_sq is None else total_sq + sq
    if small:
        cat = np.concatenate(small)
        sq = np.dot(cat, cat)
        total_sq = sq if total_sq is None else total_sq + sq
    return total_sq


def _scale_flats(flats, factor: float) -> None:
    for flat in flats:
        np.multiply(flat, flat.dtype.type(factor), out=flat)


def release_unused_blocks() -> None:
    if not using_gpu:
        return
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

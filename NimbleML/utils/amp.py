"""Mixed precision: Autocast + dynamic-loss-scale GradScaler."""
from __future__ import annotations
from contextlib import contextmanager
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np


_autocast_enabled = False
_autocast_dtype = None


def is_autocast_enabled() -> bool:
    return _autocast_enabled


def autocast_dtype():
    return _autocast_dtype


@contextmanager
def autocast(enabled: bool = True, dtype_name: str = "float16"):
    """Run enclosed ops in reduced precision when *enabled*."""
    global _autocast_enabled, _autocast_dtype
    prev_en, prev_dt = _autocast_enabled, _autocast_dtype
    if enabled:
        np_backend.set_dtype(dtype_name)
        _autocast_enabled = True
        _autocast_dtype = np_backend.dtype
    try:
        yield
    finally:
        _autocast_enabled = prev_en
        _autocast_dtype = prev_dt
        if prev_dt is not None:
            name = "float16" if "float16" in str(prev_dt) else (
                "bfloat16" if "bfloat16" in str(prev_dt) else (
                    "float64" if "float64" in str(prev_dt) else "float32"
                )
            )
            np_backend.set_dtype(name)
        else:
            np_backend.set_dtype("float32")


class GradScaler:
    """Dynamic loss scaling for fp16 training.

    Multiply the loss by ``scale`` before ``backward()`` so fp16 activation
    gradients do not underflow, divide parameter grads by ``scale`` before
    clipping/stepping, and adapt the scale: halve on overflow (non-finite grad
    norm), double after ``growth_interval`` consecutive good steps.

    The cap is 32768 — the scale itself flows through fp16 scalars in the
    autograd graph and fp16 overflows at 65504.
    """

    MAX_SCALE = 2.0**15

    def __init__(
        self,
        init_scale: float = 2.0**12,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 200,
    ):
        self.scale = float(min(init_scale, self.MAX_SCALE))
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self._growth_tracker = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_(self, params) -> None:
        """Divide grads by the current scale in place (no allocation)."""
        inv = 1.0 / self.scale
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            np.multiply(g, g.dtype.type(inv), out=g)

    def update(self, found_inf: bool) -> None:
        """Adapt the scale after a step attempt.

        Args:
            found_inf: True when the unscaled grad norm was non-finite
                (the caller must also skip ``optimizer.step()``).
        """
        if found_inf:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.MAX_SCALE)
                self._growth_tracker = 0

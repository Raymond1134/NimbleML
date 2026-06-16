"""
Linear warmup wrapper around an existing learning-rate scheduler.
"""

from __future__ import annotations
from .base import LRScheduler


class LinearWarmup(LRScheduler):
    """Linearly warm up an existing scheduler's learning rate.

    Notes
    -----
    - `LinearWarmup.step()` should be called; the wrapped `scheduler` should
      not be stepped separately (this wrapper keeps them time-synchronized).
    - Warmup multiplier ramps from `start_factor` to 1 over `warmup_steps`
      steps. With the default `start_factor=0.0`, the first warmup step uses
      multiplier = 1/warmup_steps.
    """

    def __init__(self, scheduler: LRScheduler, warmup_steps: int, start_factor: float = 0.0):
        if not isinstance(scheduler, LRScheduler):
            raise TypeError(f"scheduler must be an LRScheduler, got {type(scheduler).__name__}")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        if not (0.0 <= start_factor <= 1.0):
            raise ValueError("start_factor must be in [0, 1]")

        self.inner_scheduler = scheduler
        self.warmup_steps = int(warmup_steps)
        self.start_factor = float(start_factor)

        super().__init__(scheduler.optimizer, last_epoch=scheduler.last_epoch)

    def _warmup_multiplier(self) -> float:
        if self.last_epoch >= self.warmup_steps:
            return 1.0
        progress = (self.last_epoch + 1) / self.warmup_steps
        return self.start_factor + (1.0 - self.start_factor) * progress

    def get_lr(self):
        """Public function get_lr."""
        self.inner_scheduler.last_epoch = self.last_epoch
        inner_lrs = self.inner_scheduler.get_lr()
        scale = self._warmup_multiplier()
        return [lr * scale for lr in inner_lrs]

"""Linear warmup wrapper around an existing learning-rate scheduler."""
from __future__ import annotations
from .base import LRScheduler


class LinearWarmup(LRScheduler):
    """Linear warmup wrapper for a learning-rate scheduler.

    Applies a linearly increasing multiplier to the learning rates produced by
    another scheduler. The multiplier increases from ``start_factor`` to 1.0
    over ``warmup_steps`` steps, after which the wrapped scheduler's learning
    rates are used unchanged.

    Args:
        scheduler (LRScheduler): Scheduler to wrap.
        warmup_steps (int): Number of warmup steps.
        start_factor (float): Initial learning-rate multiplier in the range [0, 1]. Defaults to 0.0.

    Notes:
        Call :meth:`step` on this wrapper rather than on the wrapped scheduler.
        The wrapper keeps both schedulers synchronized automatically.
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

    def get_lr(self):
        """Compute learning rates for the current step.

        Returns:
            list[float]: Warmed-up learning rates for all optimizer parameter groups.
        """
        self.inner_scheduler.last_epoch = self.last_epoch
        inner_lrs = self.inner_scheduler.get_lr()
        scale = self._warmup_multiplier()
        return [lr * scale for lr in inner_lrs]

    def _warmup_multiplier(self) -> float:
        if self.last_epoch >= self.warmup_steps:
            return 1.0
        progress = (self.last_epoch + 1) / self.warmup_steps
        return self.start_factor + (1.0 - self.start_factor) * progress
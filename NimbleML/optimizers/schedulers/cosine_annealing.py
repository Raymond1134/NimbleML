"""Cosine annealing learning-rate scheduler"""
import math
from .base import LRScheduler


class CosineAnnealing(LRScheduler):
    """Cosine annealing learning-rate scheduler.

    Decays each learning rate from its initial value to ``eta_min`` following
    a cosine schedule over ``T_max`` epochs or steps.

    The learning rate for each parameter group is computed as::

        lr = eta_min + (base_lr - eta_min) *
             (1 + cos(pi * t / T_max)) / 2

    Args:
        optimizer (Optimizer): Optimizer whose learning rates will be updated.
        T_max (int): Number of epochs or steps in the cosine cycle.
        eta_min (float): Minimum learning rate reached at the end of the cycle. Defaults to 0.0.
    """

    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = T_max
        self.eta_min = float(eta_min)

    def get_lr(self):
        """Compute learning rates for the current epoch or step.

        Returns:
            list[float]: Learning rate for each optimizer parameter group.
        """
        cosine_decay = (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


"""Backward-compatible alias."""
CosineAnnealingLR = CosineAnnealing

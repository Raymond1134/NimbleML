"""Cosine annealing learning-rate scheduler"""
import math
from .base import LRScheduler


class CosineAnnealing(LRScheduler):
    """Public class CosineAnnealing."""
    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = T_max
        self.eta_min = float(eta_min)

    def get_lr(self):
        """Public function get_lr."""
        cosine_decay = (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


# Backward-compatible alias.
CosineAnnealingLR = CosineAnnealing

"""Linear decay learning-rate scheduler"""
from .base import LRScheduler


class LinearDecay(LRScheduler):
    """Linear decay learning-rate scheduler.

    Decays each learning rate linearly from its initial value to ``eta_min``
    over ``T_max`` epochs or steps, then holds it at ``eta_min``::

        lr = eta_min + (base_lr - eta_min) * max(0, 1 - t / T_max)

    Args:
        optimizer (Optimizer): Optimizer whose learning rates will be updated.
        T_max (int): Number of epochs or steps to reach ``eta_min``.
        eta_min (float): Final learning rate. Defaults to 0.0.
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
        remaining = max(0.0, 1.0 - max(0, self.last_epoch) / self.T_max)
        return [self.eta_min + (base_lr - self.eta_min) * remaining for base_lr in self.base_lrs]

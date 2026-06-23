"""Step learning rate scheduler"""
from .base import LRScheduler


class StepLR(LRScheduler):
    """Step learning-rate scheduler.

    Decays each learning rate by multiplying it by ``gamma`` every ``step_size`` epochs or steps.

    Args:
        optimizer (Optimizer): Optimizer whose learning rates will be updated.
        step_size (int): Number of epochs or steps between learning-rate updates.
        gamma (float): Multiplicative decay factor. Defaults to 0.1.
    """
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        """Compute learning rates for the current epoch or step.

        Returns:
            list[float]: Learning rate for each optimizer parameter group.
        """
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base * factor for base in self.base_lrs]

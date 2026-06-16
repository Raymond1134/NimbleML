# stepLR.py
# Step learning rate scheduler
from .base import LRScheduler


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base * factor for base in self.base_lrs]

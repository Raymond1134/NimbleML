"""Base learning-rate scheduler"""
from ..optimizer import Optimizer


class LRScheduler:
    """Base class for learning-rate schedulers.

    Subclasses implement ``get_lr()`` returning one rate per optimizer param group;
    ``step()`` applies those rates via ``optimizer.set_lr()``.
    """

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.base_lrs = list(optimizer.get_lr())
        self.last_epoch = last_epoch

    def get_lr(self):
        """Public function get_lr."""
        raise NotImplementedError

    def step(self, epoch=None):
        """Public function step."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        lrs = self.get_lr()
        if len(lrs) != len(self.base_lrs):
            raise ValueError(
                f"get_lr() returned {len(lrs)} values, expected {len(self.base_lrs)}"
            )
        self.optimizer.set_lr(lrs)

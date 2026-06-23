"""Base learning-rate scheduler"""
from ..optimizer import Optimizer


class LRScheduler:
    """Base class for learning-rate schedulers.

    A scheduler updates an optimizer's learning rate as training progresses.
    """

    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.base_lrs = list(optimizer.get_lr())
        self.last_epoch = last_epoch

    def get_lr(self):
        """Compute the current learning rates.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self, epoch=None):
        """Advance the scheduler and update optimizer learning rates.

        Args:
            epoch (int | None): Explicit epoch or step index.
            If ``None``, advances to the next epoch.
        
        Raises:
            ValueError: If the number of returned learning rates does not match the number of optimizer param groups.
        """
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

"""Stochastic gradient descent"""
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from .optimizer import Optimizer


class SGD(Optimizer):
    """Vanilla stochastic gradient descent."""
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params, learning_rate=learning_rate)

    def step(self):
        """Apply one SGD update to all parameters with gradients."""
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data -= lr * np.asarray(param.grad, dtype=np_backend.dtype)

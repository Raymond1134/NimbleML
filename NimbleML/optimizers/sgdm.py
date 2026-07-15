"""SGD with momentum"""
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from .optimizer import Optimizer


class SGDM(Optimizer):
    """SGD with momentum."""

    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocities = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]

    def step(self):
        """Apply one momentum SGD update."""
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                v = self.velocities[i]
                grad = np.asarray(param.grad, dtype=np_backend.dtype)
                np.multiply(v, self.momentum, out=v)
                v += grad
                param.data -= lr * v
            offset += len(group["params"])

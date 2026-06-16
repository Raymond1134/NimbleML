# sgdm.py
# SGD with momentum
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


class SGDM(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocities = [np.zeros(param.size, dtype=np.float64) for param in self.params]

    def step(self):
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                grad = np.asarray(param.grad, dtype=np.float64)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                param.data = np.asarray(param.data, dtype=np.float64) - lr * self.velocities[i]
            offset += len(group["params"])

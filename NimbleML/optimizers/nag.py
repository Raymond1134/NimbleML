# nag.py
# Nesterov accelerated gradient
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


class NAG(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [np.zeros(param.size, dtype=np.float64) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = np.asarray(param.grad, dtype=np.float64)
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            nesterov_grad = grad + self.momentum * self.velocities[i]
            param.data = np.asarray(param.data, dtype=np.float64) - self.learning_rate * nesterov_grad

# sgd.py
# Stochastic gradient descent
from NimbleML.utils.np_backend import np
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params, learning_rate=learning_rate)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.data -= lr * np.asarray(param.grad, dtype=np.float64)

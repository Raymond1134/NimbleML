# sgd.py
# Stochastic Gradient Descent
import numpy as np

from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            param.data -= self.learning_rate * np.asarray(param.grad, dtype=np.float64)

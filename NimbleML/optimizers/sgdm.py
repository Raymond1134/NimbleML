# sgdm.py
# Stochastic Gradient Descent with Momentum
from NimbleML.utils.tensor import Tensor
from .optimizer import Optimizer

class SGDM(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [[0.0] * param.size for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.velocities[i] = [self.momentum * vel + grad for vel, grad in zip(self.velocities[i], param.grad)]
            param.data = [val - self.learning_rate * vel for val, vel in zip(param.data, self.velocities[i])]

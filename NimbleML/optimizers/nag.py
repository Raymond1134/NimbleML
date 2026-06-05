# nag.py
# Nesterov Accelerated Gradient
from NimbleML.utils.tensor import Tensor
from .optimizer import Optimizer

class NAG(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.09):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [Tensor([0.0] * param.size, param.shape, requires_grad=False) for param in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            lookahead = param - self.learning_rate * self.momentum * self.velocities[i]
            grad = Tensor(lookahead.grad, param.shape, requires_grad=False)
            self.velocities[i] = self.momentum * self.velocities[i] - grad
            param.data = [val + self.learning_rate * vel for val, vel in zip(param.data, self.velocities[i].data)]
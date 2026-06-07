# RMSProp.py
# Root Mean Square Propagation
from .optimizer import Optimizer

class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grad_avg = [[0.0] * param.size for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.sq_grad_avg[i] = [self.rho * avg + (1 - self.rho) * grad**2 for avg, grad in zip(self.sq_grad_avg[i], param.grad)]

            param.data = [
                val - self.learning_rate * grad / (avg ** 0.5 + self.epsilon)
                for val, grad, avg in zip(param.data, param.grad, self.sq_grad_avg[i])
            ]
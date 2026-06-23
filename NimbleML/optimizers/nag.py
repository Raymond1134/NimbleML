"""Nesterov accelerated gradient"""
from NimbleML.utils.np_backend import np
from .optimizer import Optimizer


class NAG(Optimizer):
    """Nesterov accelerated gradient optimizer.

    NAG extends momentum-based gradient descent by evaluating the gradient
    using a look-ahead estimate of the parameter update, often leading to
    faster convergence than standard momentum.

    Args:
        params: Parameters to optimize.
        learning_rate (float): Learning rate. Defaults to 0.01.
        momentum (float): Momentum factor. Defaults to 0.9.
    """
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params, learning_rate=learning_rate)
        self.momentum = momentum
        self.velocities = [np.zeros(param.size, dtype=np.float64) for param in self.params]

    def step(self):
        """Perform a single optimization step.

        Updates all parameters with available gradients using Nesterov accelerated
        gradient and momentum accumulation.
        """
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                grad = np.asarray(param.grad, dtype=np.float64)
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                nesterov_grad = grad + self.momentum * self.velocities[i]
                param.data = np.asarray(param.data, dtype=np.float64) - lr * nesterov_grad
            offset += len(group["params"])

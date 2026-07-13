"""RMSProp optimizer."""
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from .optimizer import Optimizer


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, params, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(params, learning_rate=learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grad_avg = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]

    def step(self):
        """Apply one RMSProp update."""
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                sq = self.sq_grad_avg[i]
                grad = np.asarray(param.grad, dtype=np_backend.dtype)
                np.multiply(sq, self.rho, out=sq)
                sq += (1.0 - self.rho) * grad * grad
                param.data -= lr * grad / (np.sqrt(sq) + self.epsilon)
            offset += len(group["params"])

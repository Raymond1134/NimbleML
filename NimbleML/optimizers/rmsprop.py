"""RMSProp optimizer."""
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


class RMSProp(Optimizer):
    """Public class RMSProp."""
    def __init__(self, params, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(params, learning_rate=learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.sq_grad_avg = [np.zeros(param.size, dtype=np.float64) for param in self.params]

    def step(self):
        """Public function step."""
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                grad = np.asarray(param.grad, dtype=np.float64)
                self.sq_grad_avg[i] = self.rho * self.sq_grad_avg[i] + (1 - self.rho) * grad * grad
                param.data = (
                    np.asarray(param.data, dtype=np.float64)
                    - lr * grad / (np.sqrt(self.sq_grad_avg[i]) + self.epsilon)
                )
            offset += len(group["params"])

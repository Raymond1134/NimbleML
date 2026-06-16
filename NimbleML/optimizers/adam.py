# adam.py
# Adam optimizer
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]
        self.v = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        bias_corr1 = 1 - self.beta1 ** self.t
        bias_corr2 = 1 - self.beta2 ** self.t
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                grad = np.asarray(param.grad, dtype=np_backend.dtype)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
                m_hat = self.m[i] / bias_corr1
                v_hat = self.v[i] / bias_corr2
                param.data = (
                    np.asarray(param.data, dtype=np_backend.dtype)
                    - lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )
            offset += len(group["params"])

# adam.py
# Adam optimizer
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros(param.size, dtype=np.float64) for param in self.params]
        self.v = [np.zeros(param.size, dtype=np.float64) for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        bias_corr1 = 1 - self.beta1 ** self.t
        bias_corr2 = 1 - self.beta2 ** self.t
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = np.asarray(param.grad, dtype=np.float64)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2
            param.data = np.asarray(param.data, dtype=np.float64) - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# adam.py
# Adaptive Moment Estimation (Adam) optimizer
from .optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [[0.0] * param.size for param in self.params]
        self.v = [[0.0] * param.size for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        bias_corr1 = 1 - self.beta1 ** self.t
        bias_corr2 = 1 - self.beta2 ** self.t
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            self.m[i] = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m[i], grad)]
            self.v[i] = [self.beta2 * v + (1 - self.beta2) * g * g for v, g in zip(self.v[i], grad)]
            m_hat = [m / bias_corr1 for m in self.m[i]]
            v_hat = [v / bias_corr2 for v in self.v[i]]
            param.data = [val - self.learning_rate * mh / (vh ** 0.5 + self.epsilon) for val, mh, vh in zip(param.data, m_hat, v_hat)]

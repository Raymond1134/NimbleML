"""Adam and AdamW optimizers."""
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer


def _adam_update(param, grad, m, v, *, lr, beta1, beta2, bias_corr1, bias_corr2, epsilon, weight_decay):
    """Vectorized Adam(W) update for one parameter tensor."""
    grad = np.asarray(grad, dtype=np_backend.dtype)
    m *= beta1
    m += (1.0 - beta1) * grad
    v *= beta2
    v += (1.0 - beta2) * grad * grad

    m_hat = m / bias_corr1
    v_hat = v / bias_corr2
    update = lr * m_hat / (np.sqrt(v_hat) + epsilon)

    data = np.asarray(param.data, dtype=np_backend.dtype)
    if weight_decay:
        data *= 1.0 - lr * weight_decay
    data -= update
    param.data = data


class Adam(Optimizer):
    """Adam optimizer (L2-style weight decay is not applied; use AdamW for decoupled WD)."""

    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = 0.0
        self.m = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]
        self.v = [np.zeros(param.size, dtype=np_backend.dtype) for param in self.params]
        self.t = 0

    def step(self):
        """Public function step."""
        self.t += 1
        bias_corr1 = 1.0 - self.beta1 ** self.t
        bias_corr2 = 1.0 - self.beta2 ** self.t
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                i = offset + j
                _adam_update(
                    param,
                    param.grad,
                    self.m[i],
                    self.v[i],
                    lr=lr,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    bias_corr1=bias_corr1,
                    bias_corr2=bias_corr2,
                    epsilon=self.epsilon,
                    weight_decay=self.weight_decay,
                )
            offset += len(group["params"])


class AdamW(Adam):
    """Adam with decoupled weight decay (AdamW)."""

    def __init__(
        self,
        params,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    ):
        super().__init__(
            params,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        self.weight_decay = float(weight_decay)

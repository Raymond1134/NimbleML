"""Adam and AdamW optimizers."""
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np

from .optimizer import Optimizer

# Only vectorize updates for small same-shape params (biases, norm gammas).
# Batching large weight tensors would stack them into several temporary
# (n, size) buffers, a multi-hundred-MB VRAM spike each step that can overflow
# a small GPU and trigger paging. Large params are updated in place instead.
_BUCKET_MAX_ELEMS = 1 << 16  # 65,536 elements (~256 KB in float32)


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

    @staticmethod
    def _adam_update(param, grad, m, v, *, lr, beta1, beta2, bias_corr1, bias_corr2, epsilon, weight_decay):
        """In-place Adam(W) update for one parameter tensor."""
        grad = np.asarray(grad, dtype=np_backend.dtype)
        np.multiply(m, beta1, out=m)
        m += (1.0 - beta1) * grad
        np.multiply(v, beta2, out=v)
        v += (1.0 - beta2) * grad * grad

        m_hat = m / bias_corr1
        v_hat = v / bias_corr2
        update = lr * m_hat / (np.sqrt(v_hat) + epsilon)

        data = np.asarray(param.data, dtype=np_backend.dtype).reshape(-1)
        if weight_decay:
            data = data * (1.0 - lr * weight_decay) - update.reshape(-1)
        else:
            data = data - update.reshape(-1)
        param.data[...] = data

    @staticmethod
    def _adam_update_bucket(bucket, m_states, v_states, *, lr, beta1, beta2, bias_corr1, bias_corr2, epsilon, weight_decay):
        """Vectorized Adam(W) update for multiple same-sized parameters."""
        indices = [i for i, _ in bucket]
        params = [p for _, p in bucket]
        n = len(params)
        size = params[0].size

        grads = np.empty((n, size), dtype=np_backend.dtype)
        datas = np.empty((n, size), dtype=np_backend.dtype)
        for k, param in enumerate(params):
            grads[k] = np.asarray(param.grad, dtype=np_backend.dtype).reshape(-1)
            datas[k] = np.asarray(param.data, dtype=np_backend.dtype).reshape(-1)

        ms = np.stack([m_states[i] for i in indices])
        vs = np.stack([v_states[i] for i in indices])

        np.multiply(ms, beta1, out=ms)
        ms += (1.0 - beta1) * grads
        np.multiply(vs, beta2, out=vs)
        vs += (1.0 - beta2) * grads * grads

        m_hat = ms / bias_corr1
        v_hat = vs / bias_corr2
        update = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        if weight_decay:
            datas *= 1.0 - lr * weight_decay
        datas -= update

        for k, (idx, param) in enumerate(bucket):
            np.asarray(param.data, dtype=np_backend.dtype).reshape(-1)[:] = datas[k]
            m_states[idx][:] = ms[k]
            v_states[idx][:] = vs[k]

    def step(self):
        """Public function step."""
        self.t += 1
        bias_corr1 = 1.0 - self.beta1 ** self.t
        bias_corr2 = 1.0 - self.beta2 ** self.t
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            group_wd = float(group.get("weight_decay", self.weight_decay))
            active = []
            for j, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                active.append((offset + j, param))

            buckets: dict[int, list] = {}
            for idx, param in active:
                buckets.setdefault(param.size, []).append((idx, param))

            for size, bucket in buckets.items():
                if len(bucket) == 1 or size > _BUCKET_MAX_ELEMS:
                    for idx, param in bucket:
                        self._adam_update(
                            param,
                            param.grad,
                            self.m[idx],
                            self.v[idx],
                            lr=lr,
                            beta1=self.beta1,
                            beta2=self.beta2,
                            bias_corr1=bias_corr1,
                            bias_corr2=bias_corr2,
                            epsilon=self.epsilon,
                            weight_decay=group_wd,
                        )
                else:
                    self._adam_update_bucket(
                        bucket,
                        self.m,
                        self.v,
                        lr=lr,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        bias_corr1=bias_corr1,
                        bias_corr2=bias_corr2,
                        epsilon=self.epsilon,
                        weight_decay=group_wd,
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

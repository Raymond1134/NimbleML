"""Adam and AdamW with mixed-precision-safe state.

Moment buffers (and fp32 master weights for reduced-precision params) stay
in float32: in fp16, ``v = (1 - beta2) * g^2`` underflows and ``epsilon = 1e-8``
is not representable. Warmup-sized updates also round away in fp16, so
reduced-precision params keep an fp32 master copy.

GPU updates run in-place per parameter (one ElementwiseKernel each). Packing
every tensor into one buffer copies hundreds of millions of weights each step.
Checkpoint layout remains per-parameter ``m[i]``, ``v[i]``, ``master[i]``.
"""
from NimbleML.utils import np_backend
from .optimizer import Optimizer

_fused_adamw_kernel = None


def _gpu_adamw_kernel():
    global _fused_adamw_kernel
    if _fused_adamw_kernel is None:
        import cupy as cp

        _fused_adamw_kernel = cp.ElementwiseKernel(
            "float32 g, float32 m_in, float32 v_in, float32 w_in, "
            "float32 lr, float32 b1, float32 b2, float32 bc1, float32 bc2, "
            "float32 eps, float32 wd",
            "float32 m_out, float32 v_out, float32 w_out, float32 p",
            """
            float m_new = b1 * m_in + (1.0f - b1) * g;
            float v_new = b2 * v_in + (1.0f - b2) * g * g;
            float update = (m_new / bc1) / (sqrtf(v_new / bc2) + eps);
            float w_new = w_in * (1.0f - lr * wd) - lr * update;
            m_out = m_new;
            v_out = v_new;
            w_out = w_new;
            p = w_new;
            """,
            "nimbleml_adamw_step_f32",
        )
    return _fused_adamw_kernel


class Adam(Optimizer):
    """Adam. Use :class:`AdamW` for decoupled weight decay."""

    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = 0.0
        np = np_backend.np
        self._state_dtype = (
            np.float64 if np_backend.dtype == np.float64 else np.float32
        )
        self.m = [np.zeros(p.size, dtype=self._state_dtype) for p in self.params]
        self.v = [np.zeros(p.size, dtype=self._state_dtype) for p in self.params]
        self.masters = [
            np.asarray(p.data, dtype=self._state_dtype).reshape(-1).copy()
            if p.data.dtype == np.float16
            else None
            for p in self.params
        ]
        self.t = 0

    def reset_master_weights(self):
        """Re-copy fp32 masters from the current params.

        Masters are snapshotted at construction and written back on every
        ``step()``; weights loaded into the model afterwards are otherwise
        overwritten by the stale snapshot on the first update.
        """
        np = np_backend.np
        for i, p in enumerate(self.params):
            if self.masters[i] is not None:
                self.masters[i] = np.asarray(p.data, dtype=self._state_dtype).reshape(-1).copy()

    def step(self):
        self.t += 1
        bias_corr1 = 1.0 - self.beta1 ** self.t
        bias_corr2 = 1.0 - self.beta2 ** self.t
        idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            group_wd = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                if param.grad is not None:
                    self._adam_update(
                        param,
                        self.m[idx],
                        self.v[idx],
                        self.masters[idx],
                        lr=lr,
                        beta1=self.beta1,
                        beta2=self.beta2,
                        bias_corr1=bias_corr1,
                        bias_corr2=bias_corr2,
                        epsilon=self.epsilon,
                        weight_decay=group_wd,
                    )
                idx += 1
        self._notify_weights_updated()

    @staticmethod
    def _adam_update(
        param,
        m,
        v,
        master,
        *,
        lr,
        beta1,
        beta2,
        bias_corr1,
        bias_corr2,
        epsilon,
        weight_decay,
    ):
        np = np_backend.np
        state_dtype = m.dtype
        g = np.asarray(param.grad, dtype=state_dtype).reshape(-1)

        if np_backend.using_gpu:
            try:
                kernel = _gpu_adamw_kernel()
            except Exception:
                kernel = None
            w = master if master is not None else param.data.reshape(-1)
            if kernel is not None and state_dtype == np.float32 and w.dtype == np.float32:
                p_out = np.empty_like(w) if param.data.dtype != np.float32 else param.data.reshape(-1)
                f = np.float32
                kernel(
                    g, m, v, w,
                    f(lr), f(beta1), f(beta2), f(bias_corr1), f(bias_corr2),
                    f(epsilon), f(weight_decay),
                    m, v, w, p_out,
                )
                if master is not None:
                    param.data[...] = np.asarray(master, dtype=param.data.dtype).reshape(
                        param.data.shape
                    )
                elif param.data.dtype != np.float32:
                    param.data[...] = p_out.astype(param.data.dtype, copy=False).reshape(
                        param.data.shape
                    )
                return

        if not np_backend.using_gpu and state_dtype == np.dtype("float32"):
            import numpy as host_np
            from NimbleML._native_loader import native

            w = master if master is not None else param.data.reshape(-1)
            wh = host_np.ascontiguousarray(host_np.asarray(w, dtype=host_np.float32))
            gh = host_np.ascontiguousarray(host_np.asarray(g, dtype=host_np.float32))
            mh = host_np.ascontiguousarray(host_np.asarray(m, dtype=host_np.float32))
            vh = host_np.ascontiguousarray(host_np.asarray(v, dtype=host_np.float32))
            native.adamw_step(
                wh, gh, mh, vh, float(lr), float(beta1), float(beta2), float(bias_corr1),
                float(bias_corr2), float(epsilon), float(weight_decay),
            )
            if wh is not w:
                w[...] = wh
            if mh is not m:
                m[...] = mh
            if vh is not v:
                v[...] = vh
            if master is not None:
                param.data[...] = np.asarray(master, dtype=param.data.dtype).reshape(
                    param.data.shape
                )
            return

        np.multiply(m, beta1, out=m)
        m += (1.0 - beta1) * g
        np.multiply(v, beta2, out=v)
        v += (1.0 - beta2) * g * g

        update = (m / bias_corr1) / (np.sqrt(v / bias_corr2) + epsilon)
        w = master if master is not None else param.data.reshape(-1)
        if weight_decay:
            w *= 1.0 - lr * weight_decay
        w -= (lr * update).astype(state_dtype, copy=False)
        if master is not None:
            param.data[...] = np.asarray(master, dtype=param.data.dtype).reshape(
                param.data.shape
            )


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

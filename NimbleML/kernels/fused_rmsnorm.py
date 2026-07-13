"""Fused RMSNorm forward and backward on the active NumPy/CuPy backend.

RMSNorm over the last dimension:

    y = gamma * x / sqrt(mean(x^2) + epsilon)

For fp16 inputs the row statistics (``mean(x^2)`` and downstream per-row
factors) are computed in float32. GPU uses a CuPy RawKernel (one block per
row) when available; otherwise ufuncs. Native CUDA device entrypoints are
used for fp32 when ``rmsnorm_forward_device`` is present.
"""
from __future__ import annotations

from NimbleML._native_loader import native as _native  # noqa: F401  # required
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, on_device, using_gpu

_rmsnorm_fwd_raw = None
_rmsnorm_bwd_raw = None


def _cupy_rmsnorm_kernels():
    global _rmsnorm_fwd_raw, _rmsnorm_bwd_raw
    if _rmsnorm_fwd_raw is not None:
        return _rmsnorm_fwd_raw, _rmsnorm_bwd_raw
    import cupy as cp

    _rmsnorm_fwd_raw = cp.RawKernel(
        r"""
        extern "C" __global__
        void nimbleml_rmsnorm_fwd(const float* x, const float* gamma, float* out,
                                  float* ms, float* rms, int rows, int dim, float eps) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const float* xr = x + (size_t)row * dim;
            float* orow = out + (size_t)row * dim;
            __shared__ float shared[256];
            float local = 0.0f;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) {
                float v = xr[d];
                local += v * v;
            }
            shared[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
                __syncthreads();
            }
            float mean_sq = shared[0] / (float)dim;
            float r = sqrtf(mean_sq + eps);
            if (threadIdx.x == 0) { ms[row] = mean_sq; rms[row] = r; }
            __syncthreads();
            float inv = 1.0f / r;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) {
                orow[d] = xr[d] * inv * gamma[d];
            }
        }
        """,
        "nimbleml_rmsnorm_fwd",
    )
    _rmsnorm_bwd_raw = cp.RawKernel(
        r"""
        extern "C" __global__
        void nimbleml_rmsnorm_bwd(const float* grad, const float* x, const float* gamma,
                                  const float* ms, const float* rms, float* grad_x,
                                  float* grad_gamma, int rows, int dim, float eps) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const float* xr = x + (size_t)row * dim;
            const float* gr = grad + (size_t)row * dim;
            float* gxr = grad_x + (size_t)row * dim;
            float r = rms[row];
            float inv = 1.0f / r;
            float mean_sq = ms[row];
            __shared__ float shared[256];
            float local = 0.0f;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) {
                local += (gr[d] * gamma[d]) * xr[d];
            }
            shared[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
                __syncthreads();
            }
            float row_dot = shared[0];
            float grad_ms = row_dot * (-0.5f) * powf(mean_sq + eps, -1.5f);
            float coef = (2.0f / (float)dim) * grad_ms;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) {
                float x_hat = xr[d] * inv;
                atomicAdd(grad_gamma + d, gr[d] * x_hat);
                gxr[d] = gr[d] * gamma[d] * inv + xr[d] * coef;
            }
        }
        """,
        "nimbleml_rmsnorm_bwd",
    )
    return _rmsnorm_fwd_raw, _rmsnorm_bwd_raw


def _ptr(arr) -> int:
    return int(arr.data.ptr)


def _try_native_device_fwd(x_f32, g_f32, epsilon):
    if not hasattr(_native, "rmsnorm_forward_device"):
        return None
    rows, dim = int(x_f32.shape[0]), int(x_f32.shape[1])
    out = np.empty_like(x_f32)
    ms = np.empty((rows,), dtype=np.float32)
    rms = np.empty((rows,), dtype=np.float32)
    _native.rmsnorm_forward_device(
        _ptr(x_f32), _ptr(g_f32), _ptr(out), _ptr(ms), _ptr(rms), rows, dim, float(epsilon)
    )
    return out, ms, rms


def _try_native_device_bwd(grad_f32, x_f32, g_f32, ms, rms, epsilon):
    if not hasattr(_native, "rmsnorm_backward_device"):
        return None
    rows, dim = int(x_f32.shape[0]), int(x_f32.shape[1])
    gx = np.empty_like(x_f32)
    gg = np.zeros((dim,), dtype=np.float32)
    _native.rmsnorm_backward_device(
        _ptr(grad_f32), _ptr(x_f32), _ptr(g_f32), _ptr(ms), _ptr(rms),
        _ptr(gx), _ptr(gg), rows, dim, float(epsilon),
    )
    return gx, gg


def fused_rmsnorm_forward(x, gamma, epsilon=1e-5):
    x_arr = on_device(x, dtype=np_backend.dtype)
    g_arr = on_device(gamma, dtype=np_backend.dtype)
    flat = x_arr.reshape(-1, x_arr.shape[-1])
    rows, dim = int(flat.shape[0]), int(flat.shape[1])

    if using_gpu and flat.dtype == np.float32:
        xf = np.ascontiguousarray(flat)
        gf = np.ascontiguousarray(g_arr.reshape(-1))
        native_out = _try_native_device_fwd(xf, gf, epsilon)
        if native_out is not None:
            out, ms, rms = native_out
            return out.reshape(x_arr.shape), x_arr, ms.reshape(rows, 1), rms.reshape(rows, 1)
        try:
            fwd, _ = _cupy_rmsnorm_kernels()
            out = np.empty_like(xf)
            ms = np.empty((rows,), dtype=np.float32)
            rms = np.empty((rows,), dtype=np.float32)
            fwd((rows,), (256,), (xf, gf, out, ms, rms, rows, dim, np.float32(epsilon)))
            return out.reshape(x_arr.shape), x_arr, ms.reshape(rows, 1), rms.reshape(rows, 1)
        except Exception:
            pass

    if x_arr.dtype == np.float16:
        ms = np.mean(x_arr * x_arr, axis=-1, keepdims=True, dtype=np.float32)
        rms = np.sqrt(ms + np.float32(epsilon))
        inv = (1.0 / rms).astype(np.float16)
        out = (x_arr * inv) * g_arr
        return out, x_arr, ms, rms
    ms = np.mean(x_arr * x_arr, axis=-1, keepdims=True)
    rms = np.sqrt(ms + epsilon)
    out = (x_arr / rms) * g_arr
    return out, x_arr, ms, rms


def fused_rmsnorm_backward(grad_out, x, gamma, ms, rms, epsilon=1e-5):
    grad = on_device(grad_out, dtype=np_backend.dtype)
    x_arr = on_device(x, dtype=np_backend.dtype)
    g_arr = on_device(gamma, dtype=np_backend.dtype)
    ms_arr = np.asarray(ms)
    rms_arr = np.asarray(rms)
    d = x_arr.shape[-1]
    flat_x = x_arr.reshape(-1, d)
    flat_g = grad.reshape(-1, d)
    rows = int(flat_x.shape[0])

    if using_gpu and flat_x.dtype == np.float32:
        xf = np.ascontiguousarray(flat_x)
        gf = np.ascontiguousarray(g_arr.reshape(-1))
        go = np.ascontiguousarray(flat_g)
        ms1 = np.ascontiguousarray(ms_arr.reshape(-1), dtype=np.float32)
        rms1 = np.ascontiguousarray(rms_arr.reshape(-1), dtype=np.float32)
        native = _try_native_device_bwd(go, xf, gf, ms1, rms1, epsilon)
        if native is not None:
            gx, gg = native
            return gx.reshape(x_arr.shape), gg
        try:
            _, bwd = _cupy_rmsnorm_kernels()
            gx = np.empty_like(xf)
            gg = np.zeros((d,), dtype=np.float32)
            bwd((rows,), (256,), (go, xf, gf, ms1, rms1, gx, gg, rows, d, np.float32(epsilon)))
            return gx.reshape(x_arr.shape), gg
        except Exception:
            pass

    if x_arr.dtype == np.float16:
        inv = (1.0 / rms_arr).astype(np.float16)
        x_hat = x_arr * inv
        reduce_axes = tuple(range(x_arr.ndim - 1))
        grad_gamma = np.sum(grad * x_hat, axis=reduce_axes, dtype=np.float32)

        grad_x_hat = grad * g_arr
        row_dot = np.sum(grad_x_hat * x_arr, axis=-1, keepdims=True, dtype=np.float32)
        grad_ms = row_dot * np.float32(-0.5) * (ms_arr + np.float32(epsilon)) ** np.float32(-1.5)
        coef = ((2.0 / d) * grad_ms).astype(np.float16)
        grad_x = grad_x_hat * inv + x_arr * coef
        return grad_x, grad_gamma

    x_hat = x_arr / rms_arr
    reduce_axes = tuple(range(x_arr.ndim - 1))
    grad_gamma = np.sum(grad * x_hat, axis=reduce_axes)

    grad_x_hat = grad * g_arr
    row_dot = np.sum(grad_x_hat * x_arr, axis=-1, keepdims=True)
    grad_ms = row_dot * (-0.5) * (ms_arr + epsilon) ** (-1.5)
    coef = (2.0 / d) * grad_ms
    grad_x = grad_x_hat / rms_arr + x_arr * coef
    return grad_x, grad_gamma

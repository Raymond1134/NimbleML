"""Fused log-softmax + cross-entropy on the active NumPy/CuPy backend.

For fp16 inputs, reductions (``sum_exp``, per-sample loss sum) run with fp32
accumulators. GPU uses a fused RawKernel (one block per row) when available.
"""
from __future__ import annotations
from NimbleML._native_loader import native as _native  # noqa: F401
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_label_indices, np, on_device, using_gpu

_ce_fwd_f32 = None
_ce_bwd_f32 = None
_ce_fwd_f16 = None
_ce_bwd_f16 = None


def _cupy_ce_kernels():
    global _ce_fwd_f32, _ce_bwd_f32, _ce_fwd_f16, _ce_bwd_f16
    if _ce_fwd_f32 is not None:
        return _ce_fwd_f32, _ce_bwd_f32, _ce_fwd_f16, _ce_bwd_f16
    import cupy as cp

    _ce_fwd_f32 = cp.RawKernel(
        r"""
        extern "C" __global__
        void nimbleml_ce_fwd_f32(const float* logits, const long long* labels,
                                 float* max_vals, float* sum_exp, float* loss_out,
                                 int rows, int classes) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const float* xr = logits + (size_t)row * classes;
            __shared__ float shared[256];
            float local = -1e30f;
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                local = fmaxf(local, xr[d]);
            }
            shared[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
                __syncthreads();
            }
            float m = shared[0];
            float acc = 0.0f;
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                acc += expf(xr[d] - m);
            }
            shared[threadIdx.x] = acc;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                float s = shared[0];
                max_vals[row] = m;
                sum_exp[row] = s;
                long long y = labels[row];
                float correct = xr[y];
                atomicAdd(loss_out, (m + logf(s) - correct) / (float)rows);
            }
        }
        """,
        "nimbleml_ce_fwd_f32",
    )
    _ce_bwd_f32 = cp.RawKernel(
        r"""
        extern "C" __global__
        void nimbleml_ce_bwd_f32(const float* logits, const long long* labels,
                                 const float* max_vals, const float* sum_exp, float* grad,
                                 int rows, int classes, float scale) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const float* xr = logits + (size_t)row * classes;
            float* gr = grad + (size_t)row * classes;
            float m = max_vals[row];
            float inv = 1.0f / sum_exp[row];
            long long y = labels[row];
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                float p = expf(xr[d] - m) * inv;
                if (d == (int)y) p -= 1.0f;
                gr[d] = p * scale;
            }
        }
        """,
        "nimbleml_ce_bwd_f32",
    )
    _ce_fwd_f16 = cp.RawKernel(
        r"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void nimbleml_ce_fwd_f16(const __half* logits, const long long* labels,
                                 float* max_vals, float* sum_exp, float* loss_out,
                                 int rows, int classes) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const __half* xr = logits + (size_t)row * classes;
            __shared__ float shared[256];
            float local = -1e30f;
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                local = fmaxf(local, __half2float(xr[d]));
            }
            shared[threadIdx.x] = local;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
                __syncthreads();
            }
            float m = shared[0];
            float acc = 0.0f;
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                acc += expf(__half2float(xr[d]) - m);
            }
            shared[threadIdx.x] = acc;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                float s = shared[0];
                max_vals[row] = m;
                sum_exp[row] = s;
                long long y = labels[row];
                float correct = __half2float(xr[y]);
                atomicAdd(loss_out, (m + logf(s) - correct) / (float)rows);
            }
        }
        """,
        "nimbleml_ce_fwd_f16",
    )
    _ce_bwd_f16 = cp.RawKernel(
        r"""
        #include <cuda_fp16.h>
        extern "C" __global__
        void nimbleml_ce_bwd_f16(const __half* logits, const long long* labels,
                                 const float* max_vals, const float* sum_exp, __half* grad,
                                 int rows, int classes, float scale) {
            int row = blockIdx.x;
            if (row >= rows) return;
            const __half* xr = logits + (size_t)row * classes;
            __half* gr = grad + (size_t)row * classes;
            float m = max_vals[row];
            float inv = 1.0f / sum_exp[row];
            long long y = labels[row];
            for (int d = threadIdx.x; d < classes; d += blockDim.x) {
                float p = expf(__half2float(xr[d]) - m) * inv;
                if (d == (int)y) p -= 1.0f;
                gr[d] = __float2half(p * scale);
            }
        }
        """,
        "nimbleml_ce_bwd_f16",
    )
    return _ce_fwd_f32, _ce_bwd_f32, _ce_fwd_f16, _ce_bwd_f16


def _gpu_ce_forward(logits_arr, labels):
    rows, classes = int(logits_arr.shape[0]), int(logits_arr.shape[1])
    fwd_f32, _, fwd_f16, _ = _cupy_ce_kernels()
    max_vals = np.empty((rows,), dtype=np.float32)
    sum_exp = np.empty((rows,), dtype=np.float32)
    loss_out = np.zeros((1,), dtype=np.float32)
    labels_c = np.ascontiguousarray(labels)
    logits_c = np.ascontiguousarray(logits_arr)
    if logits_arr.dtype == np.float16:
        fwd_f16((rows,), (256,), (logits_c, labels_c, max_vals, sum_exp, loss_out, rows, classes))
    else:
        fwd_f32((rows,), (256,), (logits_c, labels_c, max_vals, sum_exp, loss_out, rows, classes))
    return loss_out, logits_c, max_vals.reshape(rows, 1), sum_exp.reshape(rows, 1)


def _gpu_ce_backward(grad_scale, logits_arr, labels, max_vals, sum_exp):
    rows, classes = int(logits_arr.shape[0]), int(logits_arr.shape[1])
    _, bwd_f32, _, bwd_f16 = _cupy_ce_kernels()
    scale = float(grad_scale) / float(rows)
    labels_c = np.ascontiguousarray(labels)
    logits_c = np.ascontiguousarray(logits_arr)
    max_c = np.ascontiguousarray(np.asarray(max_vals).reshape(-1), dtype=np.float32)
    sum_c = np.ascontiguousarray(np.asarray(sum_exp).reshape(-1), dtype=np.float32)
    grad = np.empty_like(logits_c)
    if logits_arr.dtype == np.float16:
        bwd_f16((rows,), (256,), (logits_c, labels_c, max_c, sum_c, grad, rows, classes, np.float32(scale)))
    else:
        bwd_f32((rows,), (256,), (logits_c, labels_c, max_c, sum_c, grad, rows, classes, np.float32(scale)))
    return grad


def fused_crossentropy_forward(logits, label_indices):
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size, _ = logits_arr.shape
    labels = as_label_indices(label_indices, batch_size=batch_size)

    if using_gpu and logits_arr.dtype in (np.float16, np.float32):
        try:
            return _gpu_ce_forward(logits_arr, labels)
        except Exception:
            pass

    acc = np.float32 if logits_arr.dtype == np.float16 else logits_arr.dtype

    max_vals = np.max(logits_arr, axis=1, keepdims=True)
    shifted = logits_arr - max_vals
    sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True, dtype=acc)
    log_sum_exp = max_vals.ravel().astype(acc, copy=False) + np.log(sum_exp.ravel())

    row_idx = np.arange(batch_size, dtype=np.int64)
    correct_logits = logits_arr[row_idx, labels]
    per_sample = log_sum_exp - correct_logits
    loss = np.sum(per_sample) / batch_size
    return loss, logits_arr, max_vals, sum_exp


def fused_crossentropy_backward(grad_scale, logits, label_indices, max_vals, sum_exp):
    logits_arr = on_device(logits, dtype=np_backend.dtype)
    batch_size, _ = logits_arr.shape
    labels = as_label_indices(label_indices, batch_size=batch_size)

    if using_gpu and logits_arr.dtype in (np.float16, np.float32):
        try:
            return _gpu_ce_backward(grad_scale, logits_arr, labels, max_vals, sum_exp)
        except Exception:
            pass

    max_arr = on_device(max_vals, dtype=logits_arr.dtype)
    sum_arr = np.asarray(sum_exp)

    shifted = logits_arr - max_arr
    probs = np.exp(shifted)
    inv_sum = (1.0 / sum_arr).astype(logits_arr.dtype, copy=False)
    probs *= inv_sum
    row_idx = np.arange(batch_size, dtype=np.int64)
    probs[row_idx, labels] -= 1.0
    probs *= logits_arr.dtype.type(float(grad_scale) / batch_size)
    return probs

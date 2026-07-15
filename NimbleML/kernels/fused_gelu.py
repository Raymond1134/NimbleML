"""Fused GELU forward/backward on the active NumPy/CuPy backend.

GPU path stays on-device (no host roundtrip). Prefers a single CuPy
``ElementwiseKernel`` launch; falls back to ufuncs. CPU uses the native extension.
Math matches ``gelu_forward_cpu`` (tanh approximation).
"""
from __future__ import annotations
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np, on_device, using_gpu

_GELU_K = 0.7978845608028654  # sqrt(2/pi)
_GELU_COEF = 0.044715
_DU_DX_COEF = 0.134145

_gelu_fwd_kernel = None
_gelu_bwd_kernel = None


def _cupy_kernels():
    global _gelu_fwd_kernel, _gelu_bwd_kernel
    if _gelu_fwd_kernel is not None:
        return _gelu_fwd_kernel, _gelu_bwd_kernel
    import cupy as cp

    # Compute in float32 internally; store out in input dtype, tanh_u in float32.
    _gelu_fwd_kernel = cp.ElementwiseKernel(
        "T x",
        "T out, float32 tu",
        """
        float xf = (float)x;
        float x3 = xf * xf * xf;
        const float k = 0.7978845608028654f;
        const float coef = 0.044715f;
        tu = tanhf(k * (xf + coef * x3));
        out = (T)(0.5f * xf * (1.0f + tu));
        """,
        "nimbleml_gelu_fwd",
    )
    _gelu_bwd_kernel = cp.ElementwiseKernel(
        "T g, T x, float32 tu",
        "T gx",
        """
        float gf = (float)g;
        float xf = (float)x;
        const float k = 0.7978845608028654f;
        const float du = 0.134145f;
        float du_dx = k * (1.0f + du * xf * xf);
        float sech2 = 1.0f - tu * tu;
        float d = 0.5f * (1.0f + tu) + 0.5f * xf * sech2 * du_dx;
        gx = (T)(gf * d);
        """,
        "nimbleml_gelu_bwd",
    )
    return _gelu_fwd_kernel, _gelu_bwd_kernel


def fused_gelu_forward(arr):
    """Apply GELU element-wise. Returns ``(out, tanh_u)`` for backward."""
    x = on_device(arr, dtype=np_backend.dtype)
    if not using_gpu:
        if x.dtype == np.float32:
            from NimbleML.kernels._native_bridge import native_gelu_forward

            return native_gelu_forward(x)
        # float64 (gradchecks) and other dtypes: full-precision NumPy math —
        # the native kernel is fp32-only and would poison fp64 gradients.
        tu = np.tanh(_GELU_K * (x + _GELU_COEF * x * x * x))
        out = 0.5 * x * (1.0 + tu)
        return out.astype(x.dtype, copy=False), tu

    try:
        fwd, _ = _cupy_kernels()
        out = np.empty_like(x)
        tu = np.empty(x.shape, dtype=np.float32)
        fwd(x, out, tu)
        return out, tu
    except Exception:
        # Ufunc fallback (still on-device).
        xf = x.astype(np.float32, copy=False)
        x3 = xf * xf * xf
        tu = np.tanh(np.float32(_GELU_K) * (xf + np.float32(_GELU_COEF) * x3))
        out = (np.float32(0.5) * xf * (np.float32(1.0) + tu)).astype(x.dtype, copy=False)
        return out, tu


def fused_gelu_backward(grad_out, arr, tanh_u=None):
    """Backpropagate through :func:`fused_gelu_forward`."""
    if tanh_u is None:
        _, tanh_u = fused_gelu_forward(arr)
    x = on_device(arr, dtype=np_backend.dtype)
    g = on_device(grad_out, dtype=np_backend.dtype)
    if not using_gpu:
        if x.dtype == np.float32:
            from NimbleML.kernels._native_bridge import native_gelu_backward

            return native_gelu_backward(g, x, tanh_u)
        tu = np.asarray(tanh_u)
        du_dx = _GELU_K * (1.0 + _DU_DX_COEF * x * x)
        sech2 = 1.0 - tu * tu
        d = 0.5 * (1.0 + tu) + 0.5 * x * sech2 * du_dx
        return (g * d).astype(x.dtype, copy=False)

    tu = on_device(tanh_u, dtype=np.float32)
    try:
        _, bwd = _cupy_kernels()
        gx = np.empty_like(x)
        bwd(g, x, tu, gx)
        return gx
    except Exception:
        xf = x.astype(np.float32, copy=False)
        gf = g.astype(np.float32, copy=False)
        du_dx = np.float32(_GELU_K) * (np.float32(1.0) + np.float32(_DU_DX_COEF) * xf * xf)
        sech2 = np.float32(1.0) - tu * tu
        d = np.float32(0.5) * (np.float32(1.0) + tu) + np.float32(0.5) * xf * sech2 * du_dx
        return (gf * d).astype(x.dtype, copy=False)

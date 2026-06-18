"""maxpool2d module for NimbleML."""
# 2D max pooling layer
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor
from ._ops import _kernel_dims


def _pool2d_windows(x, kernel_size, stride):
    kH, kW = _kernel_dims(kernel_size)
    N, C, H, W = x.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid max-pool output size: out_H={out_H}, out_W={out_W}. Check input shape, kernel_size, and stride.")

    as_strided = np.lib.stride_tricks.as_strided
    sN, sC, sH, sW = x.strides
    patch_shape = (N, C, out_H, out_W, kH, kW)
    patch_strides = (sN, sC, stride * sH, stride * sW, sH, sW)
    patches = as_strided(x, shape=patch_shape, strides=patch_strides)
    return patches, {"N": N, "C": C, "H": H, "W": W, "out_H": out_H, "out_W": out_W, "kH": kH, "kW": kW, "stride": stride}


def _scatter_patch_grads(patch_grads, meta):
    N = meta["N"]
    C = meta["C"]
    H = meta["H"]
    W = meta["W"]
    out_H = meta["out_H"]
    out_W = meta["out_W"]
    kH = meta["kH"]
    kW = meta["kW"]
    stride = meta["stride"]

    x_grad = np.zeros((N, C, H, W), dtype=np.float64)
    for oh in range(out_H):
        for ow in range(out_W):
            hs = oh * stride
            ws = ow * stride
            x_grad[:, :, hs:hs + kH, ws:ws + kW] += patch_grads[:, :, oh, ow, :, :]

    return x_grad


def _scatter_grad_to_argmax(patches_flat, argmax, grad_out):
    """Route output gradients to the max positions in each pool window."""
    window_size = patches_flat.shape[-1]
    mask = argmax[..., np.newaxis] == np.arange(window_size, dtype=argmax.dtype)
    return mask.astype(np.float64) * grad_out[..., np.newaxis]


class MaxPool2D(Module):
    """Public class MaxPool2D."""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, inputs):
        """Public function forward."""
        if inputs.ndim != 4:
            raise ValueError("MaxPool2d expects (N, C, H, W).")

        x = inputs.data.reshape(inputs.shape)
        patches, meta = _pool2d_windows(x, self.kernel_size, self.stride)
        N, C, out_H, out_W = meta["N"], meta["C"], meta["out_H"], meta["out_W"]
        kH, kW = meta["kH"], meta["kW"]

        patches_flat = patches.reshape(N, C, out_H, out_W, kH * kW)
        out_data = patches_flat.max(axis=4)
        argmax = patches_flat.argmax(axis=4)

        output = Tensor(out_data.ravel(), (N, C, out_H, out_W), requires_grad=inputs.requires_grad, _children=(inputs,), _op="maxpool2d")

        def _backward():
            if output.grad is None or not inputs.requires_grad:
                return

            grad_out = output.grad.reshape(N, C, out_H, out_W)
            grad_patches_flat = _scatter_grad_to_argmax(patches_flat, argmax, grad_out)
            grad_patches = grad_patches_flat.reshape(N, C, out_H, out_W, kH, kW)
            grad_x = _scatter_patch_grads(grad_patches, meta)
            inputs._accumulate_grad(grad_x.ravel())

        output._backward = _backward
        return output

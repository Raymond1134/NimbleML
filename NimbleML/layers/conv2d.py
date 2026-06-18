"""conv2d module for NimbleML."""
# 2D convolutional layer
from math import sqrt
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor
from ._ops import _kernel_dims


def _im2col(x, kernel_size, stride=1, padding=0):
    """Unfold (N, C, H, W) input into patch columns for matrix-multiply convolution."""
    kH, kW = _kernel_dims(kernel_size)
    N, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_pad, W_pad = x.shape[2], x.shape[3]
    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1

    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid convolution output size: out_H={out_H}, out_W={out_W}. Check input shape, kernel_size, stride, and padding.")

    as_strided = np.lib.stride_tricks.as_strided
    sN, sC, sH, sW = x.strides
    patch_shape = (N, out_H, out_W, C, kH, kW)
    patch_strides = (sN, stride * sH, stride * sW, sC, sH, sW)
    patches = as_strided(x, shape=patch_shape, strides=patch_strides)
    cols = patches.reshape(N * out_H * out_W, C * kH * kW)

    meta = {
        "N": N,
        "C": C,
        "H": H,
        "W": W,
        "out_H": out_H,
        "out_W": out_W,
        "stride": stride,
        "padding": padding,
        "kernel_size": kernel_size,
        "kH": kH,
        "kW": kW,
    }
    return cols, meta


def _col2im(cols, meta):
    """Scatter patch gradients back into (N, C, H, W) input layout."""
    N = meta["N"]
    C = meta["C"]
    H = meta["H"]
    W = meta["W"]
    out_H = meta["out_H"]
    out_W = meta["out_W"]
    stride = meta["stride"]
    padding = meta["padding"]
    kH = meta["kH"]
    kW = meta["kW"]

    patch_grads = cols.reshape(N, out_H, out_W, C, kH, kW)
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    x_grad = np.zeros((N, C, H_pad, W_pad), dtype=np.float64)

    for oh in range(out_H):
        for ow in range(out_W):
            hs = oh * stride
            ws = ow * stride
            x_grad[:, :, hs:hs + kH, ws:ws + kW] += patch_grads[:, oh, ow, :, :, :]

    if padding > 0:
        x_grad = x_grad[:, :, padding:padding + H, padding:padding + W]

    return x_grad.reshape(N, C, H, W)


class Conv2D(Module):
    """Public class Conv2D."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kH, kW = _kernel_dims(kernel_size)
        fan_in = in_channels * kH * kW
        scale = sqrt(2.0 / fan_in)
        weight_shape = (out_channels, in_channels, kH, kW)
        weight_data = np.random.uniform(-1, 1, size=weight_shape) * scale
        self.weights = Tensor(weight_data, weight_shape, requires_grad=True)
        self.biases = Tensor(np.zeros(out_channels), (out_channels,), requires_grad=True) if bias else None

    def forward(self, inputs):
        """Public function forward."""
        if inputs.ndim != 4:
            raise ValueError("Conv2D expects (N, C, H, W).")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[1]}.")

        x = inputs.data.reshape(inputs.shape)
        cols, meta = _im2col(x, self.kernel_size, self.stride, self.padding)
        W_flat = self.weights.data.reshape(self.out_channels, -1).T
        out = cols @ W_flat
        out = out.reshape(meta["N"], meta["out_H"], meta["out_W"], self.out_channels)
        out = out.transpose(0, 3, 1, 2)

        if self.biases is not None:
            out = out + self.biases.data.reshape(1, self.out_channels, 1, 1)

        children = [inputs, self.weights]
        if self.biases is not None:
            children.append(self.biases)

        output = Tensor(
            out.ravel(),
            (meta["N"], self.out_channels, meta["out_H"], meta["out_W"]),
            requires_grad=any(child.requires_grad for child in children),
            _children=tuple(children),
            _op="conv2d",
        )

        def _backward():
            if output.grad is None:
                return

            grad_out = output.grad.reshape(meta["N"], self.out_channels, meta["out_H"], meta["out_W"])
            grad_cols = grad_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

            if self.weights.requires_grad:
                grad_W_flat = cols.T @ grad_cols
                grad_W = grad_W_flat.T.reshape(self.weights.shape)
                self.weights._accumulate_grad(grad_W.ravel())

            if self.biases is not None and self.biases.requires_grad:
                grad_bias = grad_out.sum(axis=(0, 2, 3))
                self.biases._accumulate_grad(grad_bias.ravel())

            if inputs.requires_grad:
                grad_patches = grad_cols @ W_flat.T
                grad_x = _col2im(grad_patches, meta)
                inputs._accumulate_grad(grad_x.ravel())

        output._backward = _backward
        return output

    def parameters(self):
        """Public function parameters."""
        params = [self.weights]
        if self.biases is not None:
            params.append(self.biases)
        return params

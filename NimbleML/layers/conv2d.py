"""2D convolutional layer."""
from math import sqrt
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.shape import kernel_dims
from NimbleML.utils.tensor import Tensor, _grad_out


class Conv2D(Module):
    """2D convolution layer.

    Applies learnable convolutional filters over 4D input tensors of shape
    (N, C, H, W). Internally uses an im2col transformation to convert
    convolution into a matrix multiplication for efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kH, kW = kernel_dims(kernel_size)
        fan_in = in_channels * kH * kW
        scale = sqrt(2.0 / fan_in)
        weight_shape = (out_channels, in_channels, kH, kW)
        weight_data = np.random.uniform(-1, 1, size=weight_shape) * scale
        self.weights = Tensor(weight_data, weight_shape, requires_grad=True)
        self.biases = Tensor(np.zeros(out_channels), (out_channels,), requires_grad=True) if bias else None

    def forward(self, inputs):
        """Applies a 2D convolution to the input tensor.

        Args:
            inputs (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Output tensor of shape
                (N, out_channels, H_out, W_out).

        Raises:
            ValueError: If the input tensor is not of shape (N, C, H, W).
            ValueError: If the input channels do not match the expected number of channels.
        
        Examples:
            >>> layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            >>> inputs = Tensor(np.random.randn(1, 3, 28, 28), (1, 3, 28, 28))
            >>> output = layer.forward(inputs)
        """
        if inputs.ndim != 4:
            raise ValueError("Conv2D expects (N, C, H, W).")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[1]}.")

        x = inputs.data.reshape(inputs.shape)
        cols, meta = Conv2D._im2col(x, self.kernel_size, self.stride, self.padding)
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

            grad_out = _grad_out(output, (meta["N"], self.out_channels, meta["out_H"], meta["out_W"]))
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
                grad_x = Conv2D._col2im(grad_patches, meta)
                inputs._accumulate_grad(grad_x.ravel())

        output._backward = _backward
        return output

    def parameters(self):
        """Returns learnable parameters of the layer.

        Returns:
            list[Tensor]: Weights and optional bias tensors.
        
        Examples:
            >>> layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3)
            >>> params = layer.parameters()
        """
        params = [self.weights]
        if self.biases is not None:
            params.append(self.biases)
        return params

    @staticmethod
    def _im2col(x, kernel_size, stride=1, padding=0):
        kH, kW = kernel_dims(kernel_size)
        N, C, H, W = x.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        H_pad, W_pad = x.shape[2], x.shape[3]
        out_H = (H_pad - kH) // stride + 1
        out_W = (W_pad - kW) // stride + 1

        if out_H <= 0 or out_W <= 0:
            raise ValueError(
                f"Invalid convolution output size: out_H={out_H}, out_W={out_W}. "
                "Check input shape, kernel_size, stride, and padding."
            )

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

    @staticmethod
    def _col2im(cols, meta):
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
        x_grad = np.zeros((N, C, H_pad, W_pad), dtype=np_backend.dtype)

        oh_idx = np.arange(out_H)[:, None] * stride + np.arange(kH)[None, :]
        ow_idx = np.arange(out_W)[:, None] * stride + np.arange(kW)[None, :]
        n_i = np.arange(N)[:, None, None, None, None, None]
        c_i = np.arange(C)[None, None, None, :, None, None]
        oh_i = oh_idx[None, :, None, None, :, None]
        ow_i = ow_idx[None, None, :, None, None, :]
        np.add.at(x_grad, (n_i, c_i, oh_i, ow_i), patch_grads)

        if padding > 0:
            x_grad = x_grad[:, :, padding : padding + H, padding : padding + W]

        return x_grad.reshape(N, C, H, W)

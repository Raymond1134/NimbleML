"""2D max pooling layer."""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.shape import kernel_dims
from NimbleML.utils.tensor import Tensor


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("MaxPool2d expects (N, C, H, W).")

        x = inputs.data.reshape(inputs.shape)
        kH, kW = kernel_dims(self.kernel_size)
        N, C, H, W = x.shape
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1

        if out_H <= 0 or out_W <= 0:
            raise ValueError(
                f"Invalid max-pool output size: out_H={out_H}, out_W={out_W}. "
                "Check input shape, kernel_size, and stride."
            )

        as_strided = np.lib.stride_tricks.as_strided
        sN, sC, sH, sW = x.strides
        patch_shape = (N, C, out_H, out_W, kH, kW)
        patch_strides = (sN, sC, self.stride * sH, self.stride * sW, sH, sW)
        patches = as_strided(x, shape=patch_shape, strides=patch_strides)

        patches_flat = patches.reshape(N, C, out_H, out_W, kH * kW)
        out_data = patches_flat.max(axis=4)
        argmax = patches_flat.argmax(axis=4)

        output = Tensor(
            out_data.ravel(),
            (N, C, out_H, out_W),
            requires_grad=inputs.requires_grad,
            _children=(inputs,),
            _op="maxpool2d",
        )

        def _backward():
            if output.grad is None or not inputs.requires_grad:
                return

            grad_out = output.grad.reshape(N, C, out_H, out_W)
            window_size = patches_flat.shape[-1]
            mask = argmax[..., np.newaxis] == np.arange(window_size, dtype=argmax.dtype)
            grad_patches_flat = mask.astype(np.float64) * grad_out[..., np.newaxis]
            grad_patches = grad_patches_flat.reshape(N, C, out_H, out_W, kH, kW)

            x_grad = np.zeros((N, C, H, W), dtype=np.float64)
            for oh in range(out_H):
                for ow in range(out_W):
                    hs = oh * self.stride
                    ws = ow * self.stride
                    x_grad[:, :, hs : hs + kH, ws : ws + kW] += grad_patches[:, :, oh, ow, :, :]

            inputs._accumulate_grad(x_grad.ravel())

        output._backward = _backward
        return output

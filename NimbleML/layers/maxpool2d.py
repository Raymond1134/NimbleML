"""2D max pooling layer."""
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.shape import kernel_dims
from NimbleML.utils.tensor import Tensor, _grad_out


class MaxPool2D(Module):
    """2D max pooling layer.

    Downsamples spatial dimensions by selecting the maximum value within
    each pooling window. The operation is applied independently to each
    channel of the input tensor.

    Input shape: (N, C, H, W)
    Output shape: (N, C, out_H, out_W)
    """
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, inputs):
        """Applies 2D max pooling.

        For each pooling window, the maximum value is selected and propagated
        to the output. During backpropagation, gradients are routed only to
        the elements that produced the maxima.

        Args:
            inputs (Tensor): Input tensor of shape ``(N, C, H, W)``.

        Returns:
            Tensor: Pooled output tensor of shape ``(N, C, out_H, out_W)``.

        Raises:
            ValueError: If the input is not 4-dimensional.
            ValueError: If the kernel size and stride produce an invalid output shape.
        
        Examples:
            >>> layer = MaxPool2D(kernel_size=2, stride=2)
            >>> inputs = Tensor(np.random.randn(1, 3, 28, 28), (1, 3, 28, 28))
            >>> output = layer.forward(inputs)
        """
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

            grad_out = _grad_out(output, (N, C, out_H, out_W))
            window_size = patches_flat.shape[-1]
            mask = argmax[..., np.newaxis] == np.arange(window_size, dtype=argmax.dtype)
            grad_patches_flat = mask.astype(np_backend.dtype) * grad_out[..., np.newaxis]
            grad_patches = grad_patches_flat.reshape(N, C, out_H, out_W, kH, kW)

            x_grad = np.zeros((N, C, H, W), dtype=np_backend.dtype)
            oh_idx = np.arange(out_H)[:, None] * self.stride + np.arange(kH)[None, :]
            ow_idx = np.arange(out_W)[:, None] * self.stride + np.arange(kW)[None, :]
            n_i = np.arange(N)[:, None, None, None, None, None]
            c_i = np.arange(C)[None, :, None, None, None, None]
            oh_i = oh_idx[None, None, :, None, :, None]
            ow_i = ow_idx[None, None, None, :, None, :]
            np.add.at(x_grad, (n_i, c_i, oh_i, ow_i), grad_patches)

            inputs._accumulate_grad(x_grad.ravel())

        output._backward = _backward
        return output

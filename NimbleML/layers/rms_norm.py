"""Root Mean Square Layer Normalization (RMSNorm)."""
from math import prod

from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Normalizes activations using their root mean square (RMS) over the last
    dimension and applies a learnable scaling parameter ``gamma``.
    """

    def __init__(self, normalized_shape, epsilon=1e-5):
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        self.gamma = Tensor(
            np.ones(normalized_shape),
            (normalized_shape,),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Applies RMS normalization to the input tensor.

        RMSNorm computes: y = gamma * (x / RMS(x))
        where: RMS(x) = sqrt(mean(x²) + epsilon)
        and the mean is taken over the last dimension.

        Args:
            inputs (Tensor): Input tensor whose last dimension matches ``normalized_shape``.

        Returns:
            Tensor: RMS-normalized tensor with the same shape as ``inputs``.

        Raises:
            ValueError: If the last input dimension does not match ``normalized_shape``.
        
        Examples:
            >>> layer = RMSNorm(normalized_shape=10)
            >>> inputs = Tensor(np.random.randn(10, 10), (10, 10))
            >>> output = layer.forward(inputs)
        """
        if inputs.shape[-1] != self.normalized_shape:
            raise ValueError(f"Expected last dim {self.normalized_shape}, got {inputs.shape[-1]}")

        x = inputs
        gamma = self.gamma
        epsilon = self.epsilon
        in_shape = x.shape
        d = in_shape[-1]

        row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1
        x_arr = Tensor._asarray(x.data).reshape(row_count, d)
        g_arr = Tensor._asarray(gamma.data).reshape(d)

        ms = np.mean(x_arr * x_arr, axis=-1, keepdims=True)
        rms = np.sqrt(ms + epsilon)
        out_arr = ((x_arr / rms) * g_arr).reshape(in_shape)

        save_x = _save_for_backward(x_arr)
        save_ms = _save_for_backward(ms)
        save_rms = _save_for_backward(rms)
        save_g = _save_for_backward(g_arr)

        requires_grad = x.requires_grad or gamma.requires_grad
        out = Tensor(
            out_arr.ravel(),
            in_shape,
            requires_grad=requires_grad,
            _children=(x, gamma),
            _op="rms_norm",
        )

        def _backward():
            if out.grad is None:
                return

            grad_out = _grad_out(out, (row_count, d))
            if gamma.requires_grad:
                x_hat = save_x / save_rms
                grad_gamma = np.sum(grad_out * x_hat, axis=0)
                gamma._accumulate_grad(grad_gamma.ravel())

            if x.requires_grad:
                grad_x_hat = grad_out * save_g
                grad_ms = np.sum(
                    grad_x_hat * save_x * (-0.5) * (save_ms + epsilon) ** (-1.5),
                    axis=-1,
                    keepdims=True,
                )
                grad_x_from_ms = (2.0 / d) * save_x * grad_ms
                grad_x_direct = grad_x_hat / save_rms
                grad_x = grad_x_direct + grad_x_from_ms
                x._accumulate_grad(grad_x.reshape(in_shape).ravel())

        out._backward = _backward
        return out

    def parameters(self):
        """Returns learnable parameters of the layer.

        Returns:
            list[Tensor]: List containing the scale parameter ``gamma``.

        Examples:
            >>> layer = RMSNorm(normalized_shape=10)
            >>> params = layer.parameters()
        """
        return [self.gamma]

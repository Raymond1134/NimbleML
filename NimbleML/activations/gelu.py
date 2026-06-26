"""Gaussian Error Linear Unit (GELU) activation."""
from NimbleML.kernels.fused_gelu import fused_gelu_backward, fused_gelu_forward
from NimbleML.neural_network import Module


def gelu_forward(arr):
    """Computes the GELU activation using the tanh approximation.

    Args:
        arr (np.ndarray): Input array.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            A tuple containing:
            - out: GELU output.
            - tanh_u: Cached tanh(u) values used for backward computation.

    Examples:
        >>> from NimbleML.utils.np_backend import np
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> out, tanh_u = gelu_forward(arr)
    """
    return fused_gelu_forward(arr)


def gelu_backward(grad_out, arr, tanh_u=None):
    """Computes the backward pass of the GELU activation.

    Args:
        grad_out (np.ndarray): Gradient of the output.
        arr (np.ndarray): Input array.
        tanh_u (np.ndarray, optional): Cached tanh(u) values used for backward computation.

    Returns:
        np.ndarray: Gradient of the input.

    Examples:
        >>> from NimbleML.utils.np_backend import np
        >>> grad_out = np.array([1.0, 2.0, 3.0])
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> tanh_u = np.array([1.0, 2.0, 3.0])
        >>> grad_in = gelu_backward(grad_out, arr, tanh_u)
    """
    return fused_gelu_backward(grad_out, arr, tanh_u)


class Gelu(Module):
    """Gaussian Error Linear Unit (GELU) activation module.

    Applies the GELU activation function element-wise to the input tensor.
    The GELU activation is defined as: GELU(x) = x * Φ(x), where Φ(x) is
    the cumulative distribution function of the standard normal distribution.
    """

    def forward(self, inputs):
        """Applies the GELU activation function.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with GELU applied element-wise.
        """
        return inputs.gelu()

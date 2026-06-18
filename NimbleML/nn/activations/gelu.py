"""Gaussian Error Linear Unit (GELU) activation."""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np


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
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> out, tanh_u = gelu_forward(arr)
    """
    k = np.sqrt(2.0 / np.pi)
    x3 = arr * arr * arr
    u = k * (arr + 0.044715 * x3)
    tanh_u = np.tanh(u)
    out = 0.5 * arr * (1.0 + tanh_u)
    return out, tanh_u


def gelu_backward(grad_out, arr, tanh_u=None):
    """Computes the backward pass of the GELU activation.
    
    Args:
        grad_out (np.ndarray): Gradient of the output.
        arr (np.ndarray): Input array.
        tanh_u (np.ndarray, optional): Cached tanh(u) values used for backward computation.

    Returns:
        np.ndarray: Gradient of the input.
    
    Examples:
        >>> grad_out = np.array([1.0, 2.0, 3.0])
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> tanh_u = np.array([1.0, 2.0, 3.0])
        >>> grad_in = gelu_backward(grad_out, arr, tanh_u)
    """
    grad_out = np.ascontiguousarray(grad_out)
    arr = np.ascontiguousarray(arr)
    k = np.sqrt(2.0 / np.pi)
    if tanh_u is None:
        tanh_u = np.tanh(k * (arr + 0.044715 * arr * arr * arr))
    else:
        tanh_u = np.ascontiguousarray(tanh_u)
    du_dx = k * (1.0 + 0.134145 * arr * arr)
    sech2 = 1.0 - tanh_u * tanh_u
    return grad_out * (0.5 * (1.0 + tanh_u) + 0.5 * arr * sech2 * du_dx)


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

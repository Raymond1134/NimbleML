"""GELU activation (Gaussian Error Linear Unit)."""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np


def gelu_forward(arr):
    """GELU forward (tanh approximation). Returns (output, tanh_u) for backward."""
    k = np.sqrt(2.0 / np.pi)
    x3 = arr * arr * arr
    u = k * (arr + 0.044715 * x3)
    tanh_u = np.tanh(u)
    out = 0.5 * arr * (1.0 + tanh_u)
    return out, tanh_u


def gelu_backward(grad_out, arr, tanh_u):
    """GELU backward given pre-activation ``arr`` and cached ``tanh_u``."""
    k = np.sqrt(2.0 / np.pi)
    du_dx = k * (1.0 + 0.134145 * arr * arr)
    sech2 = 1.0 - tanh_u * tanh_u
    return grad_out * (0.5 * (1.0 + tanh_u) + 0.5 * arr * sech2 * du_dx)


class Gelu(Module):
    """Apply GELU element-wise via ``inputs.gelu()``."""

    def forward(self, inputs):
        """Public function forward."""
        return inputs.gelu()

"""Gaussian Error Linear Unit (GELU) activation."""
from NimbleML.neural_network import Module
from NimbleML.utils.activations import gelu_backward, gelu_forward

__all__ = ["Gelu", "gelu_forward", "gelu_backward"]


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

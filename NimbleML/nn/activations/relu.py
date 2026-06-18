"""Rectified Linear Unit (ReLU) activation."""
from NimbleML.neural_network import Module


class Relu(Module):
    """Rectified Linear Unit (ReLU) activation module.
    
    Applies the ReLU activation function element-wise to the input tensor.
    The ReLU activation is defined as: ReLU(x) = max(0, x).
    """

    def forward(self, inputs):
        """Applies the ReLU activation function.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with ReLU applied element-wise.
        """
        return inputs.relu()

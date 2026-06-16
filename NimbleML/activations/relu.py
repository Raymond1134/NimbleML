"""ReLU activation"""
from NimbleML.neural_network import Module


class Relu(Module):
    """Public class Relu."""
    def forward(self, inputs):
        """Public function forward."""
        return inputs.relu()

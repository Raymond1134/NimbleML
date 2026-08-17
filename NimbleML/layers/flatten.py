"""Flatten layer."""
from math import prod
from NimbleML.neural_network import Module


class Flatten(Module):
    """Flatten layer.

    Converts inputs of shape: (N, D1, D2, ..., Dk)
    into: (N, D1 * D2 * ... * Dk)
    while preserving the batch dimension.
    """
    def forward(self, inputs):
        if inputs.ndim < 2:
            raise ValueError("Flatten expects input with at least 2 dimensions (batch, ...).")

        batch = inputs.shape[0]
        flat_size = prod(inputs.shape[1:])
        return inputs.reshape((batch, flat_size))

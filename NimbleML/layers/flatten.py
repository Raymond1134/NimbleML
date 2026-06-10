# flatten.py
# Flatten layer: (N, C, H, W) -> (N, C*H*W)
from math import prod
from NimbleML.utils.tensor import Tensor
from NimbleML.neural_network import Module

class Flatten(Module):
    def forward(self, inputs):
        if inputs.ndim < 2:
            raise ValueError("Flatten expects input with at least 2 dimensions (batch, ...).")

        batch = inputs.shape[0]
        flat_size = prod(inputs.shape[1:])
        return inputs.reshape((batch, flat_size))

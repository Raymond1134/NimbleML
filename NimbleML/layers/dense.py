# dense.py
# Dense (fully connected) layer
from math import sqrt
from random import uniform
from NimbleML.neural_network import Module
from NimbleML.utils.tensor import Tensor


class Dense(Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None):
        self.in_features = in_features
        self.out_features = out_features
        scale = weight_scale if weight_scale is not None else 1.0 / sqrt(max(1, in_features))
        self.weights = Tensor(
            [uniform(-1, 1) * scale for _ in range(in_features * out_features)],
            (in_features, out_features),
            requires_grad=True,
        )
        self.biases = (Tensor([0.0] * out_features, (out_features,), requires_grad=True) if bias else None)

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {inputs.shape[-1]}")

        output = inputs @ self.weights
        if self.biases is not None:
            output = output + self.biases
        return output

    def parameters(self):
        params = [self.weights]
        if self.biases is not None:
            params.append(self.biases)
        return params

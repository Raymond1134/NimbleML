"""Transformer FFN: expand -> ReLU -> project back (per-token MLP)"""
from NimbleML.activations import Relu
from NimbleML.layers import Dense
from NimbleML.neural_network.module import Module


class FeedForward(Module):
    """Public class FeedForward."""
    def __init__(self, d_model, ff_mult=4):
        hidden = ff_mult * d_model
        self.dense1 = Dense(d_model, hidden)
        self.relu = Relu()
        self.dense2 = Dense(hidden, d_model)

    def forward(self, x):
        """Public function forward."""
        return self.dense2(self.relu(self.dense1(x)))

    def parameters(self):
        """Public function parameters."""
        params = []
        for layer in (self.dense1, self.dense2):
            params.extend(layer.parameters())
        return params

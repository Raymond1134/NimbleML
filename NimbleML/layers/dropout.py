"""Dropout regularization layer"""
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class Dropout(Module):
    """Public class Dropout."""
    def __init__(self, probability=0.5):
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.probability = probability
        self.training = True

    def forward(self, inputs):
        """Public function forward."""
        if not self.training or self.probability == 0:
            return inputs

        keep_prob = 1.0 - self.probability
        scale = 1.0 / keep_prob
        mask = (np.random.random(inputs.size) < keep_prob).astype(np.float64) * scale
        out_data = inputs.data * mask
        out = Tensor(
            out_data,
            inputs.shape,
            requires_grad=inputs.requires_grad,
            _children=(inputs,),
            _op="dropout",
        )

        def _backward():
            if out.grad is None or not inputs.requires_grad:
                return
            inputs._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out

    def train(self):
        """Public function train."""
        self.training = True

    def eval(self):
        """Public function eval."""
        self.training = False

# dropout.py
# Dropout regularization layer
import random

from NimbleML.utils.tensor import Tensor

class Dropout:
    def __init__(self, probability=0.5):
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.probability = probability
        self.training = True

    def forward(self, input):
        if not self.training or self.probability == 0:
            return input

        keep_prob = 1.0 - self.probability
        scale = 1.0 / keep_prob
        mask = [scale if random.random() < keep_prob else 0.0 for _ in input.data]
        out_data = [val * m for val, m in zip(input.data, mask)]
        out = Tensor(out_data, input.shape, requires_grad=input.requires_grad, _children=(input,), _op="dropout")

        def _backward():
            if out.grad is None or not input.requires_grad:
                return
            grad = [g * m for g, m in zip(out.grad, mask)]
            input._accumulate_grad(grad)

        out._backward = _backward
        return out

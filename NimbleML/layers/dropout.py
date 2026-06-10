# dropout.py
# Dropout regularization layer
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor
from NimbleML.neural_network import Module

class Dropout(Module):
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
        mask = (np.random.random(input.size) < keep_prob).astype(np.float64) * scale
        out_data = input.data * mask
        out = Tensor(out_data, input.shape, requires_grad=input.requires_grad, _children=(input,), _op="dropout")

        def _backward():
            if out.grad is None or not input.requires_grad:
                return
            input._accumulate_grad(out.grad * mask)

        out._backward = _backward
        return out
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

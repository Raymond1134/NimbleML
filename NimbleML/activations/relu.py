# relu.py
# ReLU activation
from NimbleML.neural_network import Module


class Relu(Module):
    def forward(self, inputs):
        return inputs.relu()

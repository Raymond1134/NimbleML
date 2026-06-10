# relu.py
# ReLU activation function
from NimbleML.neural_network import Module

class Relu(Module):
    def forward(self, input):
        return input.relu()
# RMSProp.py
# Root Mean Square Propagation
from NimbleML.utils.tensor import Tensor
from .optimizer import Optimizer

class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, rho=0.9, epsilon=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
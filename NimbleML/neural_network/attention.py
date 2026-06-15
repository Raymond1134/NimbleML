# attention.py
# Attention mechanism
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.activations import Softmax
from NimbleML.utils.tensor import Tensor

def make_casual_mask(seq_len):
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

class Attention(Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale == d_k ** 0.5
    
    
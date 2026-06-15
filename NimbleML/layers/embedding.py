# embedding.py
# Embedding layer (token ID lookup table)
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = Tensor(
            np.random.randn(vocab_size, embed_dim),
            (vocab_size, embed_dim),
            requires_grad=True,
        )

    def forward(self, inputs):
        ids = np.asarray(inputs, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"Token ID out of range [0, {self.vocab_size})")

        W = self.weights.data.reshape(self.vocab_size, self.embed_dim)
        out = W[ids]

        out_shape = (*np.asarray(inputs).shape, self.embed_dim)

        output = Tensor(
            out.ravel(),
            out_shape,
            requires_grad=self.weights.requires_grad,
            _children=(self.weights,),
            _op="embedding",
        )

        def _backward():
            if output.grad is None or not self.weights.requires_grad:
                return

            grad_flat = output.grad.reshape(-1, self.embed_dim)
            grad_W = np.zeros((self.vocab_size, self.embed_dim), dtype=np_backend.dtype)
            np.add.at(grad_W, ids, grad_flat)

            self.weights._accumulate_grad(grad_W.ravel())

        output._backward = _backward
        return output

    def parameters(self):
        return [self.weights]

# embedding.py
# Embedding layer
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor
from NimbleML.neural_network import Module


class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = Tensor(np.random.randn(vocab_size, embed_dim), (vocab_size, embed_dim), requires_grad=True)
    
    def forward(self, inputs):
        ids = np.asarray(inputs, dtype=np.int64).reshape(-1)
        W = self.weights.data.reshape(self.vocab_size, self.embed_dim)
        out = np.zeros((len(ids), self.embed_dim), dtype=np.float64)

        for row, token_id in enumerate(ids):
            token_id = int(token_id)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID {token_id} is out of range [0, {self.vocab_size})")
            out[row] = W[token_id]

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
            grad_W = np.zeros((self.vocab_size, self.embed_dim), dtype=np.float64)

            for row, token_id in enumerate(ids):
                grad_W[int(token_id)] += grad_flat[row]

            self.weights._accumulate_grad(grad_W.ravel())

        output._backward = _backward
        return output

    def parameters(self):
        return [self.weights]

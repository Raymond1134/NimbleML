"""Embedding layer (token ID lookup table)"""
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def _ensure_weight_grad(weights, vocab_size, embed_dim):
    if weights.grad is None:
        weights.zero_grad()
    return weights.grad.reshape(vocab_size, embed_dim)


def _scatter_embedding_grad(weights, ids, grad_flat, vocab_size, embed_dim):
    """Scatter-add embedding gradients in-place (``np.add.at`` on the grad buffer)."""
    if not weights.requires_grad or grad_flat is None:
        return
    grad_W = _ensure_weight_grad(weights, vocab_size, embed_dim)
    np.add.at(grad_W, ids, grad_flat)


def _accumulate_prefix_grad(weights, seq_len, grad_flat, vocab_size, embed_dim):
    """Position-embedding path: only rows ``0 .. seq_len-1`` receive gradients."""
    if not weights.requires_grad:
        return
    grad_W = _ensure_weight_grad(weights, vocab_size, embed_dim)
    grad_W[:seq_len] += grad_flat.reshape(seq_len, embed_dim)


class Embedding(Module):
    """Public class Embedding."""
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = Tensor(
            np.random.randn(vocab_size, embed_dim),
            (vocab_size, embed_dim),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Public function forward."""
        ids = np.asarray(inputs, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"Token ID out of range [0, {self.vocab_size})")

        W = Tensor._asarray(self.weights.data).reshape(self.vocab_size, self.embed_dim)
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
            grad_flat = Tensor._asarray(output.grad).reshape(-1, self.embed_dim)
            _scatter_embedding_grad(
                self.weights, ids, grad_flat, self.vocab_size, self.embed_dim
            )

        output._backward = _backward
        return output

    def forward_prefix(self, seq_len: int):
        """Embed consecutive IDs ``0 .. seq_len-1`` via a contiguous weight slice."""
        if seq_len < 0 or seq_len > self.vocab_size:
            raise ValueError(f"seq_len must be in [0, {self.vocab_size}), got {seq_len}.")
        if seq_len == 0:
            return Tensor([], (0, self.embed_dim), requires_grad=self.weights.requires_grad)

        W = Tensor._asarray(self.weights.data).reshape(self.vocab_size, self.embed_dim)
        out = W[:seq_len]

        output = Tensor(
            out.ravel(),
            (seq_len, self.embed_dim),
            requires_grad=self.weights.requires_grad,
            _children=(self.weights,),
            _op="embedding_prefix",
        )

        def _backward():
            if output.grad is None or not self.weights.requires_grad:
                return
            grad_flat = Tensor._asarray(output.grad).reshape(seq_len, self.embed_dim)
            _accumulate_prefix_grad(
                self.weights, seq_len, grad_flat, self.vocab_size, self.embed_dim
            )

        output._backward = _backward
        return output

    def parameters(self):
        """Public function parameters."""
        return [self.weights]

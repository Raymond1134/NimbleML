"""Embedding layer (token ID lookup table)"""
import numpy as host_np

from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_int64, np, using_gpu
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


def _ensure_weight_grad(weights, vocab_size, embed_dim):
    if weights.grad is None:
        weights.zero_grad()
    return weights.grad.reshape(vocab_size, embed_dim)


def _scatter_embedding_grad(weights, ids, grad_flat, vocab_size, embed_dim):
    """Scatter-add embedding gradients (host-side for duplicate ids; GPU-safe)."""
    if not weights.requires_grad or grad_flat is None:
        return
    grad_W = _ensure_weight_grad(weights, vocab_size, embed_dim)
    ids_h = host_np.asarray(ids, dtype=host_np.int64).reshape(-1)
    grad_h = host_np.asarray(
        grad_flat.get() if hasattr(grad_flat, "get") else grad_flat,
        dtype=host_np.float32,
    )
    if using_gpu:
        buf = grad_W.get()
        host_np.add.at(buf, ids_h, grad_h)
        grad_W[...] = np.asarray(buf, dtype=np_backend.dtype)
    else:
        np.add.at(grad_W, ids_h, grad_h)


def _accumulate_prefix_grad(weights, seq_len, grad_flat, vocab_size, embed_dim):
    """Position-embedding path: only rows ``0 .. seq_len-1`` receive gradients."""
    if not weights.requires_grad:
        return
    grad_W = _ensure_weight_grad(weights, vocab_size, embed_dim)
    grad_W[:seq_len] += grad_flat.reshape(seq_len, embed_dim)


class Embedding(Module):
    """Public class Embedding."""
    def __init__(self, vocab_size, embed_dim, init_std: float = 0.02):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # GPT-2 style small init. Critical when this table is tied to the LM head
        # (logits = act @ Wt): std=1.0 here gives logit std ~sqrt(d_model), an
        # absurd starting loss (~300 for a 16k vocab) and exploding gradients.
        self.weights = Tensor(
            np.random.randn(vocab_size, embed_dim) * init_std,
            (vocab_size, embed_dim),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Public function forward."""
        if isinstance(inputs, Tensor) and Tensor._is_int64_tensor(inputs):
            ids = np.asarray(inputs.data, dtype=np.int64).reshape(-1)
            in_shape = inputs.shape
        else:
            ids = as_int64(inputs).reshape(-1)
            in_shape = np.asarray(inputs).shape

        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"Token ID out of range [0, {self.vocab_size})")

        # Keep token ids on the host for backward; the forward pass only needs a
        # short-lived GPU copy for the lookup, and retaining CuPy id buffers in
        # closures lets the memory pool reuse them before backward runs.
        save_ids = host_np.asarray(
            ids.get() if hasattr(ids, "get") else ids,
            dtype=host_np.int64,
        ).copy()

        W = Tensor._asarray(self.weights.data).reshape(self.vocab_size, self.embed_dim)
        out = W[ids]

        out_shape = (*in_shape, self.embed_dim)

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
            grad_flat = _save_for_backward(
                Tensor._asarray(output.grad).reshape(-1, self.embed_dim)
            )
            _scatter_embedding_grad(
                self.weights, save_ids, grad_flat, self.vocab_size, self.embed_dim
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
        out = W[:seq_len].copy()

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
            grad_flat = _save_for_backward(
                Tensor._asarray(output.grad).reshape(seq_len, self.embed_dim)
            )
            _accumulate_prefix_grad(
                self.weights, seq_len, grad_flat, self.vocab_size, self.embed_dim
            )

        output._backward = _backward
        return output

    def parameters(self):
        """Public function parameters."""
        return [self.weights]

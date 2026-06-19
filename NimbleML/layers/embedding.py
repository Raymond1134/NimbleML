"""Embedding layer (token ID lookup table)."""
import numpy as host_np
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import as_int64, np, using_gpu
from NimbleML.utils.tensor import Tensor, _save_for_backward


class Embedding(Module):
    """Token embedding layer.

    Converts token IDs into dense vector representations using a learnable
    embedding matrix of shape (vocab_size, embed_dim).
    """
    def __init__(self, vocab_size, embed_dim, init_std: float = 0.02):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = Tensor(
            np.random.randn(vocab_size, embed_dim) * init_std,
            (vocab_size, embed_dim),
            requires_grad=True,
        )

    def forward(self, inputs):
        """Looks up embeddings for input token IDs.

        Args:
            inputs (Tensor or array-like): Integer token IDs.

        Returns:
            Tensor: Embedded representation of shape (..., embed_dim).
        
        Raises:
            ValueError: If the token IDs are out of range [0, vocab_size).
        
        Examples:
            >>> layer = Embedding(vocab_size=10000, embed_dim=768)
            >>> inputs = Tensor(np.array([1, 2, 3, 4, 5]), (5,))
            >>> output = layer.forward(inputs)
        """
        if isinstance(inputs, Tensor) and Tensor._is_int64_tensor(inputs):
            ids = np.asarray(inputs.data, dtype=np.int64).reshape(-1)
            in_shape = inputs.shape
        else:
            ids = as_int64(inputs).reshape(-1)
            in_shape = np.asarray(inputs).shape

        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab_size):
            raise ValueError(f"Token ID out of range [0, {self.vocab_size})")

        save_ids = host_np.asarray(ids.get() if hasattr(ids, "get") else ids, dtype=host_np.int64).copy()

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
            grad_flat = _save_for_backward(Tensor._asarray(output.grad).reshape(-1, self.embed_dim))
            self._scatter_embedding_grad(save_ids, grad_flat)

        output._backward = _backward
        return output

    def forward_prefix(self, seq_len: int):
        """Embeds a contiguous prefix of token IDs.

        Efficiently computes embeddings for token IDs in the range ``[0, seq_len)`` without indexing.

        Args:
            seq_len (int): Number of sequential token IDs to embed.

        Returns:
            Tensor: Embeddings of shape (seq_len, embed_dim).
        
        Raises:
            ValueError: If ``seq_len`` is out of range [0, vocab_size).
        
        Examples:
            >>> layer = Embedding(vocab_size=10000, embed_dim=768)
            >>> output = layer.forward_prefix(seq_len=1000)
        """
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
            grad_flat = _save_for_backward(Tensor._asarray(output.grad).reshape(seq_len, self.embed_dim))
            self._accumulate_prefix_grad(seq_len, grad_flat)

        output._backward = _backward
        return output

    def parameters(self):
        """Returns learnable parameters.

        Returns:
            list[Tensor]: List containing the embedding weight matrix.
        
        Examples:
            >>> layer = Embedding(vocab_size=10000, embed_dim=768)
            >>> params = layer.parameters()
        """
        return [self.weights]

    def _ensure_weight_grad(self):
        if self.weights.grad is None:
            self.weights.zero_grad()
        return self.weights.grad.reshape(self.vocab_size, self.embed_dim)

    def _scatter_embedding_grad(self, ids, grad_flat):
        if not self.weights.requires_grad or grad_flat is None:
            return
        grad_W = self._ensure_weight_grad()
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

    def _accumulate_prefix_grad(self, seq_len, grad_flat):
        if not self.weights.requires_grad:
            return
        grad_W = self._ensure_weight_grad()
        grad_W[:seq_len] += grad_flat.reshape(seq_len, self.embed_dim)

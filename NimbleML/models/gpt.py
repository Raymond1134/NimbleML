"""Minimal GPT-style language model."""
from math import prod

from NimbleML.layers import Embedding, RMSNorm
from NimbleML.neural_network.module import Module, Sequential
from NimbleML.neural_network.transformer import TransformerBlock
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


def tied_lm_head(x, embedding_weights):
    """Logits = x @ W.T with a single autograd node (no transpose tensor)."""
    in_shape = x.shape
    vocab_size, d_model = embedding_weights.shape
    if in_shape[-1] != d_model:
        raise ValueError(
            f"d_model mismatch: activations {in_shape[-1]}, weights {embedding_weights.shape[1]}"
        )

    row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1
    x_arr = Tensor._asarray(x.data).reshape(row_count, d_model)
    w_arr = Tensor._asarray(embedding_weights.data).reshape(vocab_size, d_model)
    w_T = np.ascontiguousarray(np.swapaxes(w_arr, -2, -1))
    out2d = np.matmul(x_arr, w_T)

    save_x = _save_for_backward(x_arr) if embedding_weights.requires_grad else None
    save_w = _save_for_backward(w_arr) if x.requires_grad else None

    out_shape = in_shape[:-1] + (vocab_size,)
    out = Tensor(
        out2d.ravel(),
        out_shape,
        requires_grad=x.requires_grad or embedding_weights.requires_grad,
        _children=(x, embedding_weights),
        _op="tied_lm_head",
    )

    def _backward():
        if out.grad is None:
            return

        grad_out = _grad_out(out, (row_count, vocab_size))
        if embedding_weights.requires_grad:
            grad_w = np.matmul(np.ascontiguousarray(np.swapaxes(grad_out, -2, -1)), save_x)
            embedding_weights._accumulate_grad(grad_w.ravel())
        if x.requires_grad:
            grad_x = np.matmul(grad_out, save_w)
            x._accumulate_grad(grad_x.reshape(in_shape).ravel())

    out._backward = _backward
    return out


class GPT(Module):
    """GPT language model with tied token embedding and LM head (GPT-2 style).

    Uses learned absolute positional embeddings. ``forward_prefix`` slices rows
    ``0 .. seq_len-1`` directly; results are cached per ``seq_len`` until
    ``clear_pos_encoding_cache()`` runs (once per optimizer step in training).
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, ff_mult=4):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.ff_mult = ff_mult

        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(max_seq_len, d_model)
        self.blocks = Sequential(*[TransformerBlock(d_model, num_heads, ff_mult) for _ in range(num_layers)])
        self.ln = RMSNorm(d_model)
        self._pos_cache: dict[int, Tensor] = {}

    def _absolute_pos_encoding(self, seq_len: int) -> Tensor:
        if seq_len not in self._pos_cache:
            self._pos_cache[seq_len] = self.pos_emb.forward_prefix(seq_len)
        return self._pos_cache[seq_len]

    def clear_pos_encoding_cache(self) -> None:
        """Invalidate cached ``pos_emb(0:seq_len)`` tensors."""
        self._pos_cache.clear()

    def forward(self, input_ids):
        """Run the model on token IDs and return logits ``(batch, seq, vocab_size)``."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (batch, seq), got shape {input_ids.shape}.")

        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")

        pos = self._absolute_pos_encoding(seq_len)
        x = self.token_emb(input_ids) + pos
        x = self.blocks(x)
        x = self.ln(x)
        return tied_lm_head(x, self.token_emb.weights)

    def parameters(self):
        """Return learnable parameters (token embedding weights appear once)."""
        params = []
        for layer in (self.token_emb, self.pos_emb, self.blocks, self.ln):
            params.extend(layer.parameters())
        return params

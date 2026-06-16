"""Minimal GPT-style language model."""
from NimbleML.layers import Embedding, RMSNorm
from NimbleML.neural_network.module import Module, Sequential
from NimbleML.neural_network.transformer import TransformerBlock
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


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

        token_ids = np.asarray(input_ids.data, dtype=np.int64).reshape(batch, seq_len)
        pos = self._absolute_pos_encoding(seq_len)
        x = self.token_emb(token_ids) + pos
        x = self.blocks(x)
        x = self.ln(x)
        return x.matmul(self.token_emb.weights.T)

    def parameters(self):
        """Return learnable parameters (token embedding weights appear once)."""
        params = []
        for layer in (self.token_emb, self.pos_emb, self.blocks, self.ln):
            params.extend(layer.parameters())
        return params

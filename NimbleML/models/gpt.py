# gpt.py
# Minimal GPT-style language model
from NimbleML.layers import Dense, Embedding, LayerNorm
from NimbleML.neural_network.module import Module, Sequential
from NimbleML.neural_network.transformer import TransformerBlock
from NimbleML.utils.np_backend import np


class GPT(Module):
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
        self.ln = LayerNorm(d_model)
        self.lm_head = Dense(d_model, vocab_size)

    def forward(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (batch, seq), got shape {input_ids.shape}.")

        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")

        token_ids = np.asarray(input_ids.data, dtype=np.int64).reshape(batch, seq_len)
        positions = np.arange(seq_len, dtype=np.int64)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)

    def parameters(self):
        params = []
        for layer in (self.token_emb, self.pos_emb, self.blocks, self.ln, self.lm_head):
            params.extend(layer.parameters())
        return params

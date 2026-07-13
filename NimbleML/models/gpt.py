"""GPT-style language model."""
from math import prod
from NimbleML.layers import Embedding, RMSNorm
from NimbleML.neural_network.module import Module, Sequential
from NimbleML.neural_network.transformer import TransformerBlock
from NimbleML.neural_network.transformer_fused import FusedGPTTrunk, FusedTransformerBlock
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out, _save_for_backward


class GPT(Module):
    """Minimal GPT-style autoregressive language model."""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_len,
        ff_mult=4,
        *,
        fused_blocks: bool = True,
        fused_trunk: bool = False,
        gradient_checkpointing: bool = False,
        use_rope: bool = False,
    ):
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        if fused_trunk and not fused_blocks:
            raise ValueError("fused_trunk requires fused_blocks=True.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.ff_mult = ff_mult
        self.fused_blocks = fused_blocks
        self.fused_trunk = fused_trunk
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_rope = bool(use_rope)
        if self.use_rope and not fused_blocks:
            raise ValueError("use_rope requires fused_blocks=True (RoPE is applied in the fused MHA path).")
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = None if use_rope else Embedding(max_seq_len, d_model)
        if use_rope:
            from NimbleML.utils.rope import build_rope_cache

            head_dim = d_model // num_heads
            self._rope_cos, self._rope_sin = build_rope_cache(max_seq_len, head_dim)
        else:
            self._rope_cos = self._rope_sin = None

        if fused_blocks:
            block_modules = [
                FusedTransformerBlock(
                    d_model, num_heads, ff_mult, gradient_checkpointing=gradient_checkpointing
                )
                for _ in range(num_layers)
            ]
            if self.use_rope:
                for block in block_modules:
                    block.mha._rope_cache = (self._rope_cos, self._rope_sin)
        else:
            block_modules = [TransformerBlock(d_model, num_heads, ff_mult) for _ in range(num_layers)]
        self.ln = RMSNorm(d_model)
        if fused_trunk:
            self.blocks = FusedGPTTrunk(block_modules, self.ln)
        else:
            self.blocks = Sequential(*block_modules)
        from NimbleML.losses import CrossEntropyLoss

        self._ce_loss = CrossEntropyLoss()

    def _hidden_states(self, input_ids):
        """Transformer output before the tied LM head."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D (batch, seq), got shape {input_ids.shape}.")

        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}")

        pos = self._absolute_pos_encoding(seq_len)
        x = self.token_emb(input_ids) + pos if pos is not None else self.token_emb(input_ids)
        if self.fused_trunk:
            return self.blocks(x)
        x = self.blocks(x)
        return self.ln(x)

    def forward(self, input_ids):
        """Runs a forward pass of the GPT model.

        Computes token logits for next-token prediction: logits = GPT(input_ids) → (batch, seq_len, vocab_size)

        The model:
            1. Embeds token IDs
            2. Adds learned positional embeddings
            3. Passes through transformer blocks
            4. Applies RMS normalization
            5. Projects to vocabulary space using tied embeddings

        Args:
            input_ids (Tensor): Integer token IDs of shape (batch, seq_len).

        Returns:
            Tensor: Logits over vocabulary with shape (batch, seq_len, vocab_size).

        Raises:
            ValueError:
                - If input_ids is not 2D.
                - If sequence length exceeds max_seq_len.
                - If hidden dimension does not match model configuration.
        
        Examples:
            >>> model = GPT(vocab_size=10000, d_model=768, num_heads=12, num_layers=12, max_seq_len=1024)
            >>> input_ids = Tensor.from_int64(np.array([[1, 2, 3, 4, 5]]), (1, 5))
            >>> logits = model(input_ids)
        """
        return self._tied_logits(self._hidden_states(input_ids))

    def compute_loss(self, input_ids, labels, ignore_index=None):
        """Training loss with fused tied CE (preferred over ``forward`` + CE).

        Prefer this over ``CrossEntropyLoss(model(input_ids), labels)`` during
        training: it fuses ``hidden @ embedding.T`` with fused cross-entropy so
        the forward pass never materializes a full ``(batch, seq, vocab)`` logits
        tensor.
        """
        hidden = self._hidden_states(input_ids)
        return self._ce_loss.forward_tied(
            hidden,
            self.token_emb.weights,
            labels,
            ignore_index=ignore_index,
        )

    def _tied_logits(self, x):
        embedding_weights = self.token_emb.weights
        in_shape = x.shape
        vocab_size, d_model = embedding_weights.shape
        if in_shape[-1] != d_model:
            raise ValueError(
                f"d_model mismatch: activations {in_shape[-1]}, weights {embedding_weights.shape[1]}"
            )

        row_count = prod(in_shape[:-1]) if len(in_shape) > 1 else 1
        x_arr = x._view((row_count, d_model))
        w_arr = embedding_weights._view((vocab_size, d_model))
        # Transposed *view*: BLAS/cuBLAS take strided operands, no copy needed.
        w_T = np.swapaxes(w_arr, -2, -1)
        out2d = np.matmul(x_arr, w_T)

        save_x = _save_for_backward(x_arr) if embedding_weights.requires_grad else None

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
                # grad_W = grad_out.T @ x  (.T is a view; no contiguous copy)
                grad_w = np.matmul(grad_out.T, save_x)
                embedding_weights._accumulate_grad(grad_w.ravel())
            if x.requires_grad:
                # out = x @ w_T  ⇒  grad_x = grad_out @ w_T.T == grad_out @ W
                grad_x = np.matmul(grad_out, w_T.T)
                x._accumulate_grad(grad_x.reshape(in_shape).ravel())

        out._backward = _backward
        return out

    def parameters(self):
        """Returns learnable parameters of the layer.

        Returns:
            list[Tensor]: Learnable parameters of the layer.
        
        Examples:
            >>> model = GPT(vocab_size=10000, d_model=768, num_heads=12, num_layers=12, max_seq_len=1024)
            >>> params = model.parameters()
        """
        params = []
        for layer in (self.token_emb, self.blocks):
            params.extend(layer.parameters())
        if self.pos_emb is not None:
            params.extend(self.pos_emb.parameters())
        if not self.fused_trunk:
            params.extend(self.ln.parameters())
        return params

    def generate(self, input_ids, **kwargs):
        """Autoregressive generation (see ``models.generate.generate``)."""
        from NimbleML.models.generate import generate as _generate

        return _generate(self, input_ids, **kwargs)

    def _absolute_pos_encoding(self, seq_len: int):
        if self.pos_emb is None:
            return None
        return self.pos_emb.forward_prefix(seq_len)

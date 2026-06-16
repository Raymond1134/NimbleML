"""Pre-norm transformer block: LN -> attention -> residual, LN -> FFN -> residual"""
from NimbleML.layers import LayerNorm
from .attention import MultiHeadAttention, causal_mask_tensor
from .feed_forward import FeedForward
from .module import Module, residual


class TransformerBlock(Module):
    """Public class TransformerBlock."""
    def __init__(self, d_model, num_heads, ff_mult=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult=ff_mult)

    def forward(self, x, mask=None):
        """Public function forward."""
        if mask is None:
            mask = causal_mask_tensor(x.shape[1])

        x = residual(x, lambda t: self.mha.forward(self.ln1(t), mask=mask))
        x = residual(x, lambda t: self.ffn.forward(self.ln2(t)))
        return x

    def parameters(self):
        """Public function parameters."""
        params = []
        for layer in (self.ln1, self.mha, self.ln2, self.ffn):
            params.extend(layer.parameters())
        return params

"""Transformer block."""
from NimbleML.layers import RMSNorm
from NimbleML.utils.mask import causal_mask_tensor
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .module import Module, residual


class TransformerBlock(Module):
    """Pre-normalization Transformer block."""
    def __init__(self, d_model, num_heads, ff_mult=4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.ln1 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult=ff_mult)

    def forward(self, x, mask=None):
        """Applies a Transformer block.

        Uses the pre-norm architecture:
    
            x = x + MultiHeadAttention(RMSNorm(x))
            x = x + FeedForward(RMSNorm(x))

        If no mask is provided, a causal mask is created so tokens can
        only attend to themselves and previous positions.

        Args:
            x (Tensor): Input tensor of shape (batch, seq, d_model).
            mask (Tensor or array-like, optional): Attention mask of shape (seq, seq). Defaults to a causal mask.

        Returns:
            Tensor: Output tensor of shape (batch, seq, d_model).
        
        Examples:
            >>> block = TransformerBlock(d_model=768, num_heads=12)
            >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3, 768), requires_grad=True)
            >>> mask = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), shape=(2, 3), requires_grad=True)
            >>> output = block(x, mask)
        """
        if mask is None:
            mask = causal_mask_tensor(x.shape[1])

        x = residual(x, lambda t: self.mha.forward(self.ln1(t), mask=mask))
        x = residual(x, lambda t: self.ffn.forward(self.ln2(t)))
        return x

    def parameters(self):
        """Returns all learnable parameters in the transformer block."""
        params = []
        for layer in (self.ln1, self.mha, self.ln2, self.ffn):
            params.extend(layer.parameters())
        return params

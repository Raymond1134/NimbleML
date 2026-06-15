from .module import Module, Sequential, residual

__all__ = [
    "Attention",
    "FeedForward",
    "Module",
    "MultiHeadAttention",
    "Sequential",
    "TransformerBlock",
    "make_causal_mask",
    "residual",
]


def __getattr__(name):
    if name == "FeedForward":
        from .feed_forward import FeedForward

        return FeedForward
    if name == "Attention":
        from .attention import Attention

        return Attention
    if name == "MultiHeadAttention":
        from .attention import MultiHeadAttention

        return MultiHeadAttention
    if name == "make_causal_mask":
        from .attention import make_causal_mask

        return make_causal_mask
    if name == "TransformerBlock":
        from .transformer import TransformerBlock

        return TransformerBlock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

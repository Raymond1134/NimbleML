from .module import Module, Sequential, residual

__all__ = [
    "Attention",
    "FeedForward",
    "FusedGPTTrunk",
    "FusedTransformerBlock",
    "Module",
    "MultiHeadAttention",
    "Sequential",
    "TransformerBlock",
    "make_causal_mask",
    "residual",
]

_LAZY = {
    "FeedForward": (".feed_forward", "FeedForward"),
    "Attention": (".attention", "Attention"),
    "MultiHeadAttention": (".attention", "MultiHeadAttention"),
    "make_causal_mask": ("NimbleML.utils.mask", "make_causal_mask"),
    "TransformerBlock": (".transformer", "TransformerBlock"),
    "FusedTransformerBlock": (".transformer_fused", "FusedTransformerBlock"),
    "FusedGPTTrunk": (".transformer_fused", "FusedGPTTrunk"),
}


def __getattr__(name):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    mod = importlib.import_module(module_name, __name__)
    return getattr(mod, attr)

"""Shared activation forward/backward ops used across layers and fused blocks."""
from NimbleML.kernels.fused_gelu import fused_gelu_backward, fused_gelu_forward
from NimbleML.utils.softmax import softmax_backward, softmax_forward

gelu_forward = fused_gelu_forward
gelu_backward = fused_gelu_backward

__all__ = ["gelu_forward", "gelu_backward", "softmax_forward", "softmax_backward"]

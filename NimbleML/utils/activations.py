"""Shared activation forward/backward ops used across layers, attention, and fused blocks.

These are pure array-level helpers (no autograd ``Tensor`` nodes). The GELU
helpers re-export the fused kernel so all callers share one import surface.
"""
from NimbleML.kernels.fused_gelu import fused_gelu_backward, fused_gelu_forward
from NimbleML.utils.axis import normalize_axis
from NimbleML.utils.np_backend import np

# Re-exported under the historical names so existing call sites keep working.
gelu_forward = fused_gelu_forward
gelu_backward = fused_gelu_backward


def softmax_forward(arr, axis: int = -1):
    """Computes softmax along a given axis.

    Args:
        arr (np.ndarray): Input logits array.
        axis (int): Axis along which to apply softmax.

    Returns:
        np.ndarray: Softmax probabilities with the same shape as input.
    """
    axis = normalize_axis(arr.ndim, axis)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    exps = np.exp(arr - max_vals)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def softmax_backward(grad_out, probs, axis: int = -1):
    """Computes the backward pass of softmax with respect to its input.

    Args:
        grad_out (np.ndarray): Upstream gradient of the output.
        probs (np.ndarray): Softmax output probabilities from forward pass.
        axis (int): Axis along which softmax was applied.

    Returns:
        np.ndarray: Gradient with respect to the input logits.
    """
    axis = normalize_axis(probs.ndim, axis)
    dot = np.sum(grad_out * probs, axis=axis, keepdims=True)
    return probs * (grad_out - dot)


__all__ = ["gelu_forward", "gelu_backward", "softmax_forward", "softmax_backward"]

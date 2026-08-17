"""Softmax forward/backward on the active NumPy/CuPy backend."""
from NimbleML.utils.axis import normalize_axis
from NimbleML.utils.np_backend import np


def softmax_forward(arr, axis: int = -1):
    axis = normalize_axis(arr.ndim, axis)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    exps = np.exp(arr - max_vals)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def softmax_backward(grad_out, probs, axis: int = -1):
    axis = normalize_axis(probs.ndim, axis)
    dot = np.sum(grad_out * probs, axis=axis, keepdims=True)
    return probs * (grad_out - dot)

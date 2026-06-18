"""Softmax activation with arbitrary-axis support."""
from NimbleML.neural_network import Module
from NimbleML.utils.axis import normalize_axis
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


def softmax_forward(arr, axis: int = -1):
    """Computes softmax along a given axis.

    Args:
        arr (np.ndarray): Input logits array.
        axis (int): Axis along which to apply softmax.

    Returns:
        np.ndarray: Softmax probabilities with the same shape as input.
    
    Examples:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> softmax_forward(arr, axis=0)
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
    
    Examples:
        >>> grad_out = np.array([1.0, 2.0, 3.0])
        >>> probs = np.array([0.1, 0.2, 0.3])
        >>> softmax_backward(grad_out, probs, axis=0)
    """
    axis = normalize_axis(probs.ndim, axis)
    dot = np.sum(grad_out * probs, axis=axis, keepdims=True)
    return probs * (grad_out - dot)


class Softmax(Module):
    """Softmax activation module.
    
    Applies the softmax function along a specified axis, converting
    logits into probabilities that sum to 1 along that axis.
    The softmax function is defined as:
    
        softmax(x_i) = exp(x_i) / sum_j exp(x_j).

    Default axis=-1 is commonly used for logits in classification and attention mechanisms.
    """

    def __init__(self, axis: int = -1):
        self.axis = int(axis)

    def forward(self, inputs):
        """Apply softmax along ``self.axis``.

        Args:
            inputs: Logits tensor with ``ndim >= 1``.

        Returns:
            Tensor of probabilities with the same shape as ``inputs``.

        Raises:
            ValueError: If inputs has less than 1 dimension.
        """
        if inputs.ndim < 1:
            raise ValueError("Softmax expects at least a 1D tensor.")

        in_shape = inputs.shape
        axis = normalize_axis(inputs.ndim, self.axis)
        logits = Tensor._asarray(inputs.data).reshape(in_shape)
        probs = softmax_forward(logits, axis)

        output = Tensor(
            probs.ravel(),
            in_shape,
            requires_grad=inputs.requires_grad,
            _children=(inputs,),
            _op="softmax",
        )

        def _backward():
            if output.grad is None or not inputs.requires_grad:
                return
            grad_out = output.grad.reshape(in_shape)
            grad_in = softmax_backward(grad_out, probs, axis)
            inputs._accumulate_grad(grad_in.ravel())

        output._backward = _backward
        return output

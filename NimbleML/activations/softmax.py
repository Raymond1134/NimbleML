"""Softmax activation with arbitrary-axis support."""
from NimbleML.neural_network import Module
from NimbleML.utils.activations import softmax_backward, softmax_forward
from NimbleML.utils.axis import normalize_axis
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out

__all__ = ["Softmax", "softmax_forward", "softmax_backward"]


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
        logits = inputs._view(in_shape)
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
            grad_out = _grad_out(output, in_shape)
            grad_in = softmax_backward(grad_out, probs, axis)
            inputs._accumulate_grad(grad_in.ravel())

        output._backward = _backward
        return output

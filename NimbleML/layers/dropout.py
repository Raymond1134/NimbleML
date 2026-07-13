"""Dropout regularization layer"""
from NimbleML.neural_network import Module
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _grad_out


class Dropout(Module):
    """Dropout regularization layer.

    During training, randomly sets input elements to zero with probability ``probability``
    and scales the remaining activations by ``1 / (1 - probability)``. During evaluation,
    inputs are returned unchanged.
    """
    def __init__(self, probability=0.5):
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.probability = probability
        self.training = True

    def forward(self, inputs):
        """Apply dropout to the input tensor during training; identity in eval."""
        if not self.training or self.probability == 0:
            return inputs

        keep_prob = 1.0 - self.probability
        scale = 1.0 / keep_prob
        mask = (np.random.random(inputs.size) < keep_prob).astype(np_backend.dtype) * scale
        out_data = inputs.data * mask
        out = Tensor(
            out_data,
            inputs.shape,
            requires_grad=inputs.requires_grad,
            _children=(inputs,),
            _op="dropout",
        )

        def _backward():
            if out.grad is None or not inputs.requires_grad:
                return
            grad_out = _grad_out(out, inputs.shape)
            inputs._accumulate_grad(grad_out * mask)

        out._backward = _backward
        return out

    def parameters(self):
        return []

# softmax.py
# Softmax activation (1D+ tensors, softmax over last axis)
from math import prod

from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, inputs):
        if self.axis != -1:
            raise NotImplementedError("Only axis=-1 is supported.")

        in_shape = inputs.shape
        if inputs.ndim == 0:
            raise ValueError("Softmax expects at least a 1D tensor.")

        last_dim = in_shape[-1]
        row_count = prod(in_shape[:-1]) if inputs.ndim > 1 else 1

        logits = Tensor._asarray(inputs.data).reshape(row_count, last_dim)
        max_vals = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - max_vals)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

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
            grad_out = output.grad.reshape(row_count, last_dim)
            dot = np.sum(grad_out * probs, axis=1, keepdims=True)
            grad_in = probs * (grad_out - dot)
            inputs._accumulate_grad(grad_in.ravel())

        output._backward = _backward
        return output

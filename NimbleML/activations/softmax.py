# softmax.py
# Softmax activation (1D or 2D tensors)
from NimbleML.neural_network import Module
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class Softmax(Module):
    def forward(self, inputs):
        if inputs.ndim == 1:
            logits = inputs.data.reshape(1, inputs.shape[0])
            batch_size = 1
            class_count = inputs.shape[0]
            out_shape = inputs.shape
        elif inputs.ndim == 2:
            batch_size, class_count = inputs.shape
            logits = inputs.data.reshape(inputs.shape)
            out_shape = inputs.shape
        else:
            raise ValueError("Softmax expects a 1D or 2D tensor.")

        max_vals = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - max_vals)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        output = Tensor(
            probs.ravel(),
            out_shape,
            requires_grad=inputs.requires_grad,
            _children=(inputs,),
            _op="softmax",
        )

        def _backward():
            if output.grad is None or not inputs.requires_grad:
                return
            grad_out = output.grad.reshape(batch_size, class_count)
            dot = np.sum(grad_out * probs, axis=1, keepdims=True)
            grad_in = probs * (grad_out - dot)
            inputs._accumulate_grad(grad_in.ravel())

        output._backward = _backward
        return output

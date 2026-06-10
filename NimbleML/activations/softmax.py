# softmax.py
# Softmax activation function (supports 1D or 2D tensors)
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor
from NimbleML.neural_network import Module

class Softmax(Module):
    def forward(self, input):
        if input.ndim == 1:
            logits = input.data.reshape(1, input.shape[0])
            batch_size = 1
            class_count = input.shape[0]
            out_shape = input.shape
        elif input.ndim == 2:
            batch_size, class_count = input.shape
            logits = input.data.reshape(input.shape)
            out_shape = input.shape
        else:
            raise ValueError("Softmax expects a 1D or 2D tensor.")

        max_vals = np.max(logits, axis=1, keepdims=True)
        exps = np.exp(logits - max_vals)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        output = Tensor(
            probs.ravel(),
            out_shape,
            requires_grad=input.requires_grad,
            _children=(input,),
            _op="softmax",
        )

        def _backward():
            if output.grad is None or not input.requires_grad:
                return
            grad_out = output.grad.reshape(batch_size, class_count)
            dot = np.sum(grad_out * probs, axis=1, keepdims=True)
            grad_in = probs * (grad_out - dot)
            input._accumulate_grad(grad_in.ravel())

        output._backward = _backward
        return output

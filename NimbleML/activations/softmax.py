# softmax.py
# Softmax activation function (supports 1D or 2D tensors)
from math import exp
from NimbleML.utils.tensor import Tensor


class Softmax:
    def forward(self, input):
        if input.ndim == 1:
            max_val = max(input.data)
            exps = [exp(value - max_val) for value in input.data]
            total = sum(exps)
            output = [e / total for e in exps]
            return Tensor(output, input.shape)

        if input.ndim != 2:
            raise ValueError("Softmax expects a 1D or 2D tensor.")

        batch, classes = input.shape
        output = []
        data = input.data
        for i in range(batch):
            row = data[i * classes:(i + 1) * classes]
            max_val = max(row)
            exps = [exp(value - max_val) for value in row]
            total = sum(exps)
            output.extend([e / total for e in exps])

        return Tensor(output, input.shape)
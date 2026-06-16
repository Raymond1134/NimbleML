"""Base class for all neural network modules"""


class Module:
    """Public class Module."""
    def forward(self, x):
        """Public function forward."""
        raise NotImplementedError("Not implemented yet")

    def parameters(self):
        """Public function parameters."""
        return []

    def train(self):
        """Public function train."""
        pass

    def eval(self):
        """Public function eval."""
        pass

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    """Public class Sequential."""
    def __init__(self, *layers):
        self.layers = list(layers)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, data):
        """Public function forward."""
        for layer in self.layers:
            data = layer(data)
        return data

    def parameters(self):
        """Public function parameters."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def train(self):
        """Public function train."""
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        """Public function eval."""
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()


def residual(x, sublayer):
    """Skip connection: output = x + sublayer(x). Preserves shape; gradients flow through x."""
    return x + sublayer(x)

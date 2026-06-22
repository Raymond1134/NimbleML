"""Base class for all neural network modules."""


class Module:
    """Base class for all neural network modules."""

    def forward(self, x):
        """Computes the forward pass of the module.

        Must be implemented by all subclasses.
        """
        raise NotImplementedError("Not implemented yet")

    def parameters(self):
        """Returns all learnable parameters in the module."""
        return []

    def train(self):
        """Sets module to training mode.

        Used for layers that behave differently during training (e.g., dropout, batch norm).
        """
        pass

    def eval(self):
        """Sets module to evaluation mode.

        Used to disable training-specific behavior.
        """
        pass

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    """A container that applies layers sequentially."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, data):
        """Applies a sequence of modules in order.

        Args:
            data: Input tensor.

        Returns:
            Output after passing through all layers sequentially.
        
        Examples:
            >>> sequential = Sequential(Dense(10, 20), Dense(20, 30))
            >>> data = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3), requires_grad=True)
            >>> output = sequential(data)
        """
        for layer in self.layers:
            data = layer(data)
        return data

    def parameters(self):
        """Returns all parameters from contained layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def train(self):
        """Sets all submodules to training mode."""
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        """Sets all submodules to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()


def residual(x, sublayer):
    """Applies a residual (skip) connection.

    Computes: output = x + sublayer(x)

    Args:
        x: Input tensor.
        sublayer (callable): Function or module applied to x.

    Returns:
        Tensor: Residual-connected output with same shape as x.
    
    Examples:
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), shape=(2, 3), requires_grad=True)
        >>> sublayer = Dense(3, 3)
        >>> output = residual(x, sublayer)
    """
    return x + sublayer(x)

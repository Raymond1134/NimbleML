"""Base class for all neural network modules."""


class Module:
    """Base class for all neural network modules."""

    training = True

    def forward(self, x):
        """Compute the forward pass. Subclasses must override."""
        raise NotImplementedError("Not implemented yet")

    def parameters(self):
        """Return all learnable parameters in this module (and children)."""
        return []
    
    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients on all parameters of this module."""
        for param in self.parameters():
            param.zero_grad(set_to_none=set_to_none)

    def _child_modules(self):
        """Yield direct child :class:`Module` instances."""
        for name, value in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield item

    def train(self):
        """Set this module and all children to training mode."""
        self.training = True
        for child in self._child_modules():
            child.train()

    def eval(self):
        """Set this module and all children to evaluation mode."""
        self.training = False
        for child in self._child_modules():
            child.eval()

    def __call__(self, x):
        return self.forward(x)


class ModuleList(Module):
    """Ordered list of modules (like ``nn.ModuleList``).

    Registers children for ``train`` / ``eval`` / ``parameters`` without
    applying them sequentially (use :class:`Sequential` for that).
    """

    def __init__(self, modules=None):
        self.modules = list(modules) if modules is not None else []

    def __iter__(self):
        return iter(self.modules)

    def __len__(self):
        return len(self.modules)

    def __getitem__(self, index):
        return self.modules[index]

    def __setitem__(self, index, module):
        self.modules[index] = module

    def append(self, module):
        self.modules.append(module)

    def extend(self, modules):
        self.modules.extend(modules)

    def _child_modules(self):
        for module in self.modules:
            if isinstance(module, Module):
                yield module

    def parameters(self):
        params = []
        for module in self.modules:
            if hasattr(module, "parameters"):
                params.extend(module.parameters())
        return params

    def forward(self, x):
        raise NotImplementedError(
            "ModuleList is a container only; iterate modules or use Sequential."
        )


class Sequential(Module):
    """A container that applies layers sequentially."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, data):
        """Apply each contained module in order.

        Args:
            data: Input tensor.

        Returns:
            Output after passing through all layers sequentially.
        """
        for layer in self.layers:
            data = layer(data)
        return data

    def parameters(self):
        """Return all parameters from contained layers."""
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params


def residual(x, sublayer):
    """Apply a residual (skip) connection: ``x + sublayer(x)``.

    Args:
        x: Input tensor.
        sublayer (callable): Function or module applied to ``x``.

    Returns:
        Residual-connected output with the same shape as ``x``.
    """
    return x + sublayer(x)

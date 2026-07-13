"""Helpers for running models built as Module/Sequential or legacy layer lists."""


def forward(model, data):
    """Run a forward pass on a ``Module`` or a legacy list of layers."""
    if not isinstance(model, list):
        return model(data)
    for layer in model:
        data = layer(data)
    return data


def parameters(model):
    """Collect learnable parameters from a ``Module`` or legacy layer list."""
    if not isinstance(model, list):
        return model.parameters()
    params = []
    for layer in model:
        if hasattr(layer, "parameters"):
            params.extend(layer.parameters())
    return params


def train(model):
    """Put a ``Module`` or legacy layer list into training mode."""
    if not isinstance(model, list):
        model.train()
        return
    for layer in model:
        if hasattr(layer, "train"):
            layer.train()


def eval(model):
    """Put a ``Module`` or legacy layer list into evaluation mode."""
    if not isinstance(model, list):
        model.eval()
        return
    for layer in model:
        if hasattr(layer, "eval"):
            layer.eval()

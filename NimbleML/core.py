# core.py
# Helpers for running models built as Module/Sequential or legacy layer lists


def forward(model, data):
    if not isinstance(model, list):
        return model(data)
    for layer in model:
        data = layer(data)
    return data


def parameters(model):
    if not isinstance(model, list):
        return model.parameters()
    params = []
    for layer in model:
        if hasattr(layer, "parameters"):
            params.extend(layer.parameters())
    return params


def train(model):
    if not isinstance(model, list):
        model.train()
        return
    for layer in model:
        if hasattr(layer, "train"):
            layer.train()


def eval(model):
    if not isinstance(model, list):
        model.eval()
        return
    for layer in model:
        if hasattr(layer, "eval"):
            layer.eval()

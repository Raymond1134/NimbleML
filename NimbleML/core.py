def forward(model, data):
    for layer in model:
        data = layer.forward(data)
    return data


def parameters(model):
    params = []
    for layer in model:
        if hasattr(layer, "parameters"):
            params.extend(layer.parameters())
    return params
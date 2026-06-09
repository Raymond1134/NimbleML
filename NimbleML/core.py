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

def train(model):
    for layer in model:
        if hasattr(layer, "training"):
            layer.training = True

def eval(model):
    for layer in model:
        if hasattr(layer, "training"):
            layer.training = False
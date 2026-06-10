# module.py
# Base class for all neural network modules

class Module:
    def forward(self, x):
        raise NotImplementedError("Not implemented yet")
    
    def parameters(self):
        return []
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, data):
        for layer in self.layers:
            data = layer(data)
        return data
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
    
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
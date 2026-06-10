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

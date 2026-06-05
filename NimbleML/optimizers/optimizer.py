# optimizer.py
# Base class for optimizers

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        raise NotImplementedError("Optimizer.step must be implemented by subclasses.")

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

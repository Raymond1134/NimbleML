# sgd.py
# Stochastic Gradient Descent

class SGD:
    def __init__(self, params, learning_rate=0.01):
        self.params = list(params)
        self.learning_rate = learning_rate
    
    def step(self):
        for param in self.params:
            if param.grad is None: continue
            param.data = [val - self.learning_rate * grad for val, grad in zip(param.data, param.grad)]

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

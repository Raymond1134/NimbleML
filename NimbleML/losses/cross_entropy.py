# cross_entropy.py
# Cross-Entropy Loss function

import math
from NimbleML.utils.tensor import Tensor

class CrossEntropyLoss:
    def __call__(self, logits, labels):
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        if logits.ndim == 1:
            batch_size = 1
            class_count = logits.shape[0]
            logits_data = logits.data
            label_list = [labels] if isinstance(labels, int) else labels
        elif logits.ndim == 2:
            batch_size, class_count = logits.shape
            logits_data = logits.data
            label_list = labels
        else:
            raise ValueError("CrossEntropyLoss expects 1D or 2D logits.")
        
        if len(label_list) != batch_size:
            raise ValueError("Number of labels must equal batch size.")
        
        probabilities = []
        total_loss = 0.0
        
        for i in range(batch_size):
            row = logits_data[i*class_count : (i + 1)*class_count]
            max_val = max(row)
            exps = [math.exp(val - max_val) for val in row]
            total = sum(exps)
            row_probabilities = [e / total for e in exps]
            probabilities.extend(row_probabilities)
            
            label = label_list[i]
            probability = max(row_probabilities[label], 1e-12)
            total_loss += -math.log(probability)
        
        loss = total_loss / batch_size
        output = Tensor([loss], (), requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy",)














        def _backward():
            if not logits.requires_grad:
                return

            if batch_size == 1:
                grad = list(probabilities)
                grad[label_list[0]] -= 1.0
                logits._accumulate_grad(grad)
                return

            grad = list(probabilities)
            for i in range(batch_size):
                grad[i * class_count + label_list[i]] -= 1.0

            scale = 1.0 / batch_size
            grad = [g * scale for g in grad]
            logits._accumulate_grad(grad)

        output._backward = _backward
        return output
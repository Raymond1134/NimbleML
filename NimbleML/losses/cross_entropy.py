# cross_entropy.py
# Cross-entropy loss
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class CrossEntropyLoss:
    def __call__(self, logits, labels):
        return self.forward(logits, labels)

    def forward(self, logits, labels):
        if logits.ndim == 1:
            batch_size = 1
            class_count = logits.shape[0]
            logits_arr = logits.data.reshape(1, class_count)
            label_list = [labels] if isinstance(labels, int) else labels
        elif logits.ndim == 2:
            batch_size, class_count = logits.shape
            logits_arr = logits.data.reshape(batch_size, class_count)
            label_list = labels
        else:
            raise ValueError("CrossEntropyLoss expects 1D or 2D logits.")

        if len(label_list) != batch_size:
            raise ValueError("Number of labels must equal batch size.")

        max_vals = np.max(logits_arr, axis=1, keepdims=True)
        exps = np.exp(logits_arr - max_vals)
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)

        label_indices = np.asarray(label_list, dtype=np.int64)
        correct_probs = probabilities[np.arange(batch_size), label_indices]
        loss = float(-np.sum(np.log(np.maximum(correct_probs, 1e-12))) / batch_size)

        output = Tensor(
            [loss],
            (),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="cross_entropy",
        )

        def _backward():
            if not logits.requires_grad:
                return

            grad = probabilities.copy()
            grad[np.arange(batch_size), label_indices] -= 1.0
            grad /= batch_size
            logits._accumulate_grad(grad.ravel())

        output._backward = _backward
        return output

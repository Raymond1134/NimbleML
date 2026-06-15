# cross_entropy.py
# Cross-entropy loss (1D, 2D, or 3D sequence logits)
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor


class CrossEntropyLoss:
    def __call__(self, logits, labels, ignore_index=None):
        return self.forward(logits, labels, ignore_index=ignore_index)

    def _flatten_labels(self, labels, batch_size):
        if isinstance(labels, Tensor):
            label_arr = np.asarray(labels.data, dtype=np.int64).reshape(-1)
        elif isinstance(labels, (list, tuple)):
            label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        else:
            label_arr = np.asarray(labels, dtype=np.int64).reshape(-1)

        if label_arr.size != batch_size:
            raise ValueError(
                f"Number of labels ({label_arr.size}) must equal batch size ({batch_size})."
            )
        return label_arr

    def forward(self, logits, labels, ignore_index=None):
        out_shape = logits.shape

        if logits.ndim == 1:
            flat_batch = 1
            class_count = logits.shape[0]
            logits_arr = logits.data.reshape(1, class_count)
        elif logits.ndim == 2:
            flat_batch, class_count = logits.shape
            logits_arr = logits.data.reshape(flat_batch, class_count)
        elif logits.ndim == 3:
            flat_batch = logits.shape[0] * logits.shape[1]
            class_count = logits.shape[2]
            logits_arr = logits.data.reshape(flat_batch, class_count)
        else:
            raise ValueError("CrossEntropyLoss expects 1D, 2D, or 3D logits.")

        label_indices = self._flatten_labels(labels, flat_batch)
        valid = np.ones(flat_batch, dtype=bool)
        if ignore_index is not None:
            valid = label_indices != ignore_index
            if not np.any(valid):
                return Tensor([0.0], (), requires_grad=logits.requires_grad)
            logits_arr = logits_arr[valid]
            label_indices = label_indices[valid]
            flat_batch = int(label_indices.size)

        max_vals = np.max(logits_arr, axis=1, keepdims=True)
        exps = np.exp(logits_arr - max_vals)
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)

        correct_probs = probabilities[np.arange(flat_batch), label_indices]
        loss = float(-np.sum(np.log(np.maximum(correct_probs, 1e-12))) / flat_batch)

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
            grad[np.arange(flat_batch), label_indices] -= 1.0
            grad /= flat_batch

            if ignore_index is not None:
                full_grad = np.zeros((int(np.prod(out_shape[:-1])), class_count), dtype=np_backend.dtype)
                full_grad[valid] = grad
                grad = full_grad.reshape(out_shape) if logits.ndim == 3 else full_grad.reshape(out_shape)
            elif logits.ndim == 3:
                grad = grad.reshape(out_shape)
            elif logits.ndim == 1:
                grad = grad.reshape(out_shape)

            logits._accumulate_grad(grad.ravel())

        output._backward = _backward
        return output

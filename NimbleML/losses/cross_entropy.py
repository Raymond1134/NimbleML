"""Cross-entropy loss (1D, 2D, or 3D sequence logits)"""
import numpy as host_np

from NimbleML.activations.softmax import softmax_forward
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


def _softmax_rows(logits_arr):
    return softmax_forward(logits_arr, axis=1)


def _log_softmax_cross_entropy(logits_arr, label_indices):
    """Mean negative log-likelihood without storing softmax probabilities."""
    max_vals = np.max(logits_arr, axis=1, keepdims=True)
    shifted = logits_arr - max_vals
    log_sum_exp = max_vals.ravel() + np.log(np.sum(np.exp(shifted), axis=1))
    correct_logits = logits_arr[np.arange(label_indices.size), label_indices]
    per_sample = log_sum_exp - correct_logits
    return float(np.sum(per_sample) / label_indices.size)


def _cross_entropy_logits_grad(logits_arr, label_indices, flat_batch):
    """Gradient w.r.t. logits: (softmax - one_hot) / batch_size."""
    probs = _softmax_rows(logits_arr)
    grad = probs.copy()
    grad[np.arange(flat_batch), label_indices] -= 1.0
    grad /= flat_batch
    return grad


class CrossEntropyLoss:
    """Fused softmax + log + mean cross-entropy for 1D/2D/3D logits."""

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
        """Public function forward."""
        out_shape = logits.shape

        if logits.ndim == 1:
            total_batch = 1
            class_count = logits.shape[0]
            logits_arr = Tensor._asarray(logits.data).reshape(1, class_count)
        elif logits.ndim == 2:
            total_batch, class_count = logits.shape
            logits_arr = Tensor._asarray(logits.data).reshape(total_batch, class_count)
        elif logits.ndim == 3:
            total_batch = logits.shape[0] * logits.shape[1]
            class_count = logits.shape[2]
            logits_arr = Tensor._asarray(logits.data).reshape(total_batch, class_count)
        else:
            raise ValueError("CrossEntropyLoss expects 1D, 2D, or 3D logits.")

        label_indices = self._flatten_labels(labels, total_batch)
        label_indices_host = host_np.asarray(
            label_indices.get() if hasattr(label_indices, "get") else label_indices,
            dtype=host_np.int64,
        ).copy()

        valid = np.ones(total_batch, dtype=bool)
        if ignore_index is not None:
            valid = label_indices_host != ignore_index
            if not np.any(valid):
                return Tensor([0.0], (), requires_grad=logits.requires_grad)
            logits_arr = logits_arr[valid]
            label_indices_host = label_indices_host[valid]

        flat_batch = int(label_indices_host.size)
        loss = _log_softmax_cross_entropy(logits_arr, label_indices_host)
        saved_logits = _save_for_backward(
            Tensor._asarray(logits.data).reshape(total_batch, class_count)
        )

        output = Tensor(
            [loss],
            (),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="cross_entropy",
        )

        valid_mask = valid

        def _backward():
            if output.grad is None or not logits.requires_grad:
                return

            grad_scale = float(Tensor._asarray(output.grad).reshape(-1)[0])
            full_logits = saved_logits
            if ignore_index is not None:
                active_logits = full_logits[valid_mask]
            else:
                active_logits = full_logits

            grad = _cross_entropy_logits_grad(active_logits, label_indices_host, flat_batch)
            grad *= grad_scale

            if ignore_index is not None:
                full_grad = np.zeros((total_batch, class_count), dtype=np_backend.dtype)
                full_grad[valid_mask] = grad
            else:
                full_grad = grad

            if logits.ndim == 3:
                full_grad = full_grad.reshape(out_shape)
            elif logits.ndim == 1:
                full_grad = full_grad.reshape(out_shape)

            logits._accumulate_grad(full_grad.ravel())

        output._backward = _backward
        return output

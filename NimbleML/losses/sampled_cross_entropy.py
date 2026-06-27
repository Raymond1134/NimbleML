"""Sampled cross-entropy loss for large-vocabulary language modeling."""
from __future__ import annotations
import numpy as host_np
from NimbleML.kernels.sampled_softmax import (
    sample_negative_indices,
    sampled_softmax_backward,
    sampled_softmax_forward,
)
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


class SampledCrossEntropyLoss:
    """Cross-entropy over a sampled subset of classes per row.

    Each row uses the true label plus ``num_samples`` randomly drawn negative
    classes. Use this for large ``vocab_size`` training where full softmax is
    too expensive. When ``num_samples == vocab_size - 1``, the loss matches
    full-vocabulary cross-entropy (up to sampling of negatives).
    """

    def __call__(self, logits, labels, num_samples=32, negative_indices=None, rng=None, ignore_index=None):
        return self.forward(
            logits,
            labels,
            num_samples=num_samples,
            negative_indices=negative_indices,
            rng=rng,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits,
        labels,
        *,
        num_samples=32,
        negative_indices=None,
        rng=None,
        ignore_index=None,
    ):
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
            raise ValueError("SampledCrossEntropyLoss expects 1D, 2D, or 3D logits.")

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
        if negative_indices is None:
            negatives = sample_negative_indices(class_count, label_indices_host, num_samples, rng=rng)
        else:
            negatives = host_np.asarray(negative_indices, dtype=host_np.int64)
            if negatives.shape != (flat_batch, num_samples):
                raise ValueError(
                    f"negative_indices must have shape ({flat_batch}, {num_samples}), got {negatives.shape}."
                )

        loss, _, candidate_ids, max_vals, sum_exp = sampled_softmax_forward(
            logits_arr,
            label_indices_host,
            negatives,
        )
        saved_logits = _save_for_backward(
            Tensor._asarray(logits.data).reshape(total_batch, class_count)
        )
        saved_candidates = _save_for_backward(candidate_ids)
        saved_max = _save_for_backward(max_vals)
        saved_sum_exp = _save_for_backward(sum_exp)

        output = Tensor(
            [loss],
            (),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="sampled_cross_entropy",
        )
        valid_mask = valid

        def _backward():
            if output.grad is None or not logits.requires_grad:
                return

            grad_scale = float(Tensor._asarray(output.grad).reshape(-1)[0])
            if ignore_index is not None:
                active_logits = saved_logits[valid_mask]
                active_candidates = saved_candidates
                active_max = saved_max
                active_sum_exp = saved_sum_exp
            else:
                active_logits = saved_logits
                active_candidates = saved_candidates
                active_max = saved_max
                active_sum_exp = saved_sum_exp

            grad = sampled_softmax_backward(
                grad_scale,
                active_logits,
                active_candidates,
                active_max,
                active_sum_exp,
            )

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

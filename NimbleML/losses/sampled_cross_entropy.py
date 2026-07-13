"""Sampled cross-entropy loss for large-vocabulary language modeling."""
from __future__ import annotations

import numpy as host_np

from NimbleML.kernels.sampled_softmax import (
    sample_negative_indices,
    sampled_softmax_backward,
    sampled_softmax_forward,
)
from NimbleML.losses._ce_prepare import (
    filter_ignore_index,
    flatten_logits,
    labels_to_device,
    reshape_grad_to_logits,
    scatter_row_grads,
)
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
        logits_arr, total_batch, class_count, out_shape = flatten_logits(logits)
        label_indices = labels_to_device(labels, total_batch)
        logits_arr, label_indices, valid_mask, empty = filter_ignore_index(
            logits_arr,
            label_indices,
            ignore_index,
            empty_requires_grad=logits.requires_grad,
        )
        if empty is not None:
            return empty

        flat_batch = int(label_indices.size)
        # Sampling helpers currently expect host int64 rows.
        labels_host = host_np.asarray(
            label_indices.get() if hasattr(label_indices, "get") else label_indices,
            dtype=host_np.int64,
        )
        if negative_indices is None:
            negatives = sample_negative_indices(class_count, labels_host, num_samples, rng=rng)
        else:
            negatives = host_np.asarray(negative_indices, dtype=host_np.int64)
            if negatives.shape != (flat_batch, num_samples):
                raise ValueError(
                    f"negative_indices must have shape ({flat_batch}, {num_samples}), got {negatives.shape}."
                )

        loss, _, candidate_ids, max_vals, sum_exp = sampled_softmax_forward(
            logits_arr,
            labels_host,
            negatives,
        )
        saved_logits = _save_for_backward(logits._view((total_batch, class_count)))
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

        def _backward():
            if output.grad is None or not logits.requires_grad:
                return

            grad_scale = float(output.grad[0])
            active_logits = saved_logits[valid_mask] if ignore_index is not None else saved_logits

            grad = sampled_softmax_backward(
                grad_scale,
                active_logits,
                saved_candidates,
                saved_max,
                saved_sum_exp,
            )
            full_grad = scatter_row_grads(
                grad,
                valid_mask,
                total_batch=total_batch,
                class_count=class_count,
                ignore_index=ignore_index,
            )
            full_grad = reshape_grad_to_logits(full_grad, logits, out_shape)
            logits._accumulate_grad(full_grad.ravel())

        output._backward = _backward
        return output

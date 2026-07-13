"""Cross-entropy loss for 1D, 2D, or 3D sequence logits."""
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.kernels.fused_tied_crossentropy import fused_tied_crossentropy_backward, fused_tied_crossentropy_forward
from NimbleML.losses._ce_prepare import (
    filter_ignore_index,
    flatten_logits,
    labels_to_device,
    reshape_grad_to_logits,
    scatter_row_grads,
)
from NimbleML.utils import np_backend
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification.

    Computes a softmax and negative log-likelihood loss from
    raw logits. Supports 1D, 2D, and 3D logits.

    Shapes:
        - 1D logits: ``(classes,)``
        - 2D logits: ``(batch_size, classes)``
        - 3D logits: ``(batch_size, sequence_length, classes)``
    """

    def __call__(self, logits, labels, ignore_index=None):
        return self.forward(logits, labels, ignore_index=ignore_index)

    def forward(self, logits, labels, ignore_index=None):
        """Compute the cross-entropy loss.

        Supports classification and sequence modeling logits.

        Args:
            logits (Tensor): Input logits with shape:
                - ``(classes,)``
                - ``(batch_size, classes)``
                - ``(batch_size, sequence_length, classes)``
            labels: Target class indices.
            ignore_index (int, optional): Label value to ignore when computing the loss and gradients.

        Returns:
            Tensor: Scalar loss tensor.

        Raises:
            ValueError: If logits are not 1D, 2D, or 3D.
        
        Examples:
            >>> loss_fn = CrossEntropyLoss()
            >>> logits = Tensor(np.array([0.1, 0.2, 0.3]), (3,), requires_grad=True)
            >>> out = loss_fn(logits, 0)
            >>> out.shape
            ()
        """
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

        loss, _, max_vals, sum_exp = fused_crossentropy_forward(logits_arr, label_indices)
        saved_logits = _save_for_backward(logits._view((total_batch, class_count)))
        saved_max = _save_for_backward(max_vals)
        saved_sum_exp = _save_for_backward(sum_exp)
        saved_labels = label_indices

        output = Tensor(
            [loss],
            (),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="cross_entropy",
        )

        def _backward():
            if output.grad is None or not logits.requires_grad:
                return

            grad_scale = float(output.grad[0])
            active_logits = saved_logits[valid_mask] if ignore_index is not None else saved_logits

            grad = fused_crossentropy_backward(
                grad_scale,
                active_logits,
                saved_labels,
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

    def forward_tied(self, hidden, embedding_weights, labels, ignore_index=None):
        """Cross-entropy with tied ``hidden @ embedding_weights.T`` logits.

        Fuses the LM projection and fused CE into one autograd node.
        """
        if hidden.ndim != 3:
            raise ValueError("forward_tied expects hidden of shape (batch, seq, d_model).")

        in_shape = hidden.shape
        batch, seq_len, d_model = in_shape
        row_count = batch * seq_len
        hidden_arr = hidden._view((row_count, d_model))
        weight_arr = embedding_weights._view()

        # Keep labels on-device when possible (avoid GPU↔CPU sync every step).
        if ignore_index is None:
            from NimbleML.utils.np_backend import as_label_indices

            label_indices = as_label_indices(labels, batch_size=row_count)
            valid_mask = None
            loss, save_h, save_w, saved_logits, max_vals, sum_exp = fused_tied_crossentropy_forward(
                hidden_arr,
                weight_arr,
                label_indices,
            )
        else:
            label_indices = labels_to_device(labels, row_count)
            hidden_arr, label_indices, valid_mask, empty = filter_ignore_index(
                hidden_arr,
                label_indices,
                ignore_index,
                empty_requires_grad=hidden.requires_grad or embedding_weights.requires_grad,
            )
            if empty is not None:
                return empty
            loss, save_h, save_w, saved_logits, max_vals, sum_exp = fused_tied_crossentropy_forward(
                hidden_arr,
                weight_arr,
                label_indices,
            )

        save_h = _save_for_backward(save_h)
        save_w = _save_for_backward(save_w, tensor=embedding_weights)
        save_logits = _save_for_backward(saved_logits)
        saved_max = _save_for_backward(max_vals)
        saved_sum_exp = _save_for_backward(sum_exp)
        # Capture labels for backward (device or host ndarray).
        saved_labels = label_indices

        output = Tensor(
            [loss],
            (),
            requires_grad=hidden.requires_grad or embedding_weights.requires_grad,
            _children=(hidden, embedding_weights),
            _op="tied_cross_entropy",
        )

        def _backward():
            if output.grad is None:
                return

            grad_scale = float(output.grad[0])
            grad_h, grad_w = fused_tied_crossentropy_backward(
                grad_scale,
                save_h,
                save_w,
                saved_labels,
                save_logits,
                saved_max,
                saved_sum_exp,
            )

            if ignore_index is not None:
                full_grad_h = np.zeros((row_count, d_model), dtype=np_backend.dtype)
                full_grad_h[valid_mask] = grad_h
                grad_h = full_grad_h

            if embedding_weights.requires_grad:
                embedding_weights._accumulate_grad(grad_w.ravel())
            if hidden.requires_grad:
                hidden._accumulate_grad(grad_h.reshape(in_shape).ravel())

        output._backward = _backward
        return output

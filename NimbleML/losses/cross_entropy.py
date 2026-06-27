"""Cross-entropy loss for 1D, 2D, or 3D sequence logits."""
import numpy as host_np
from NimbleML.kernels.fused_crossentropy import fused_crossentropy_backward, fused_crossentropy_forward
from NimbleML.kernels.fused_tied_crossentropy import fused_tied_crossentropy_backward, fused_tied_crossentropy_forward
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
        loss, _, max_vals, sum_exp = fused_crossentropy_forward(logits_arr, label_indices_host)
        saved_logits = _save_for_backward(
            Tensor._asarray(logits.data).reshape(total_batch, class_count)
        )
        saved_max = _save_for_backward(max_vals)
        saved_sum_exp = _save_for_backward(sum_exp)

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
                active_max = saved_max
                active_sum_exp = saved_sum_exp
            else:
                active_logits = full_logits
                active_max = saved_max
                active_sum_exp = saved_sum_exp

            grad = fused_crossentropy_backward(
                grad_scale,
                active_logits,
                label_indices_host,
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

    def forward_tied(self, hidden, embedding_weights, labels, ignore_index=None):
        """Cross-entropy with tied ``hidden @ embedding_weights.T`` logits.

        Fuses the LM projection and fused CE into one autograd node.
        """
        if hidden.ndim != 3:
            raise ValueError("forward_tied expects hidden of shape (batch, seq, d_model).")

        in_shape = hidden.shape
        batch, seq_len, d_model = in_shape
        row_count = batch * seq_len
        hidden_arr = Tensor._asarray(hidden.data).reshape(row_count, d_model)
        weight_arr = Tensor._asarray(embedding_weights.data).reshape(embedding_weights.shape)

        label_indices = self._flatten_labels(labels, row_count)
        label_indices_host = host_np.asarray(
            label_indices.get() if hasattr(label_indices, "get") else label_indices,
            dtype=host_np.int64,
        ).copy()

        valid = np.ones(row_count, dtype=bool)
        if ignore_index is not None:
            valid = label_indices_host != ignore_index
            if not np.any(valid):
                return Tensor([0.0], (), requires_grad=hidden.requires_grad or embedding_weights.requires_grad)
            hidden_arr = hidden_arr[valid]
            label_indices_host = label_indices_host[valid]

        flat_batch = int(label_indices_host.size)
        loss, save_h, save_w, max_vals, sum_exp = fused_tied_crossentropy_forward(
            hidden_arr,
            weight_arr,
            label_indices_host,
        )
        save_h = _save_for_backward(save_h)
        save_w = _save_for_backward(save_w, tensor=embedding_weights)
        saved_max = _save_for_backward(max_vals)
        saved_sum_exp = _save_for_backward(sum_exp)

        output = Tensor(
            [loss],
            (),
            requires_grad=hidden.requires_grad or embedding_weights.requires_grad,
            _children=(hidden, embedding_weights),
            _op="tied_cross_entropy",
        )
        valid_mask = valid

        def _backward():
            if output.grad is None:
                return

            grad_scale = float(Tensor._asarray(output.grad).reshape(-1)[0])
            grad_h, grad_w = fused_tied_crossentropy_backward(
                grad_scale,
                save_h,
                save_w,
                label_indices_host,
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

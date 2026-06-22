"""Regression losses with autograd support."""
from NimbleML.utils.np_backend import np
from NimbleML.utils.tensor import Tensor, _save_for_backward


class MSELoss:
    """Mean Squared Error (MSE) loss."""

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        """Computes mean squared error loss.

        Computes the elementwise mean of squared differences: loss = mean((pred - target)²)

        Args:
            pred (Tensor): Predicted values with gradients enabled.
            target (Tensor or array-like): Ground-truth values. Treated as constant (no gradients computed).

        Returns:
            Tensor: Scalar loss tensor with gradient support.
        
        Raises:
            TypeError: If pred is not a Tensor.
        
        Examples:
            >>> loss = MSELoss()
            >>> pred = Tensor(np.array([1.0, 2.0, 3.0]), shape=(3,), requires_grad=True)
            >>> target = Tensor(np.array([1.0, 2.0, 3.0]), shape=(3,), requires_grad=False)
            >>> print(loss(pred, target))
        """
        if not isinstance(pred, Tensor):
            raise TypeError("pred must be a Tensor.")
        target_arr = _target_array(target, pred.shape)
        target_tensor = Tensor(target_arr.ravel(), pred.shape, requires_grad=False)
        diff = pred - target_tensor
        return (diff * diff).mean()


class L1Loss:
    """Mean Absolute Error (MAE / L1) loss."""

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        """Computes mean absolute error loss.

        Computes the elementwise mean of absolute differences: loss = mean(|pred - target|)
        Gradient is undefined at exactly zero; this implementation uses sign(pred - target) as a subgradient.

        Args:
            pred (Tensor): Predicted values with gradients enabled.
            target (Tensor or array-like): Ground-truth values. Treated as constant (no gradients computed).

        Returns:
            Tensor: Scalar loss tensor with gradient support.
        
        Raises:
            TypeError: If pred is not a Tensor.
        
        Examples:
            >>> loss = L1Loss()
            >>> pred = Tensor(np.array([1.0, 2.0, 3.0]), shape=(3,), requires_grad=True)
            >>> target = Tensor(np.array([1.0, 2.0, 3.0]), shape=(3,), requires_grad=False)
            >>> print(loss(pred, target))
        """
        if not isinstance(pred, Tensor):
            raise TypeError("pred must be a Tensor.")

        pred_arr = Tensor._asarray(pred.data).reshape(pred.shape)
        target_arr = _target_array(target, pred.shape)
        diff = _save_for_backward(pred_arr - target_arr)
        loss = float(np.mean(np.abs(diff)))

        output = Tensor(
            [loss],
            (),
            requires_grad=pred.requires_grad,
            _children=(pred,),
            _op="l1",
        )

        def _backward():
            if output.grad is None or not pred.requires_grad:
                return
            grad_scale = float(Tensor._asarray(output.grad).reshape(-1)[0])
            grad = grad_scale * np.sign(diff) / pred.size
            pred._accumulate_grad(grad.ravel())

        output._backward = _backward
        return output


def _target_array(target, shape):
    if isinstance(target, Tensor):
        target_arr = Tensor._asarray(target.data).reshape(target.shape)
    else:
        target_arr = Tensor._asarray(target).reshape(shape)
    if target_arr.shape != shape:
        raise ValueError("pred and target must have matching shapes.")
    return target_arr

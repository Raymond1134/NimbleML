"""Regression metrics."""
from __future__ import annotations
from NimbleML.utils.np_backend import np


def mean_squared_error(y_true, y_pred) -> float:
    """Evaluation metric: mean squared error between predictions and targets.

    For training with autograd, use :class:`~NimbleML.losses.MSELoss` instead.

    MSE measures the average squared difference between predicted values and
    true values. Larger errors are penalized more heavily because the
    differences are squared.

    Args:
        y_true (array-like): Ground-truth target values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean squared error. Lower values indicate better predictions,
        with 0.0 representing a perfect fit.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not have matching shapes.

    Examples:
        >>> mse = mean_squared_error([1, 2, 3], [1, 2, 4])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def mean_absolute_error(y_true, y_pred) -> float:
    """Evaluation metric: mean absolute error between predictions and targets.

    For training with autograd, use :class:`~NimbleML.losses.L1Loss` instead.

    MAE measures the average absolute difference between predicted values and
    true values. Unlike MSE, all errors contribute linearly, making MAE less
    sensitive to large outliers.

    Args:
        y_true (array-like): Ground-truth target values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean absolute error. Lower values indicate better predictions,
        with 0.0 representing a perfect fit.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not have matching shapes.

    Examples:
        >>> mae = mean_absolute_error([1, 2, 3], [1, 2, 4])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Computes the coefficient of determination (R²).

    R² measures how well a regression model explains the variance in the
    target values compared to simply predicting the mean of the targets,
    where:
        - 1.0: Perfect predictions.
        - 0.0: Performs no better than predicting the mean target value.
        - < 0.0: Performs worse than predicting the mean.

    Args:
        y_true (array-like): Ground-truth target values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R² score.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` do not have matching shapes.

    Examples:
        >>> r2 = r2_score([1, 2, 3], [1, 2, 3])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

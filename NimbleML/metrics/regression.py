"""Regression metrics."""

from __future__ import annotations

from NimbleML.utils.np_backend import np


def mean_squared_error(y_true, y_pred) -> float:
    """Mean squared error between predictions and targets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error between predictions and targets."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination (R2)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

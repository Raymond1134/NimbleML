"""Classification metrics."""
from __future__ import annotations
from NimbleML.utils.np_backend import np


def accuracy_score(y_true, y_pred) -> float:
    """Computes classification accuracy.

    Accuracy is the fraction of predictions that match the corresponding ground-truth labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Classification accuracy in the range [0, 1].

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` contain different numbers of samples.

    Examples:
        >>> acc = accuracy_score([1, 0, 1, 1], [1, 1, 1, 0])
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, positive_label=1):
    """Computes binary precision, recall, and F1 score.

    Precision measures the fraction of predicted positives that are correct.
    Recall measures the fraction of actual positives that are correctly identified.
    F1 is the harmonic mean of precision and recall.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        positive_label: Label value treated as the positive class.

    Returns:
        tuple[float, float, float]:
            A tuple containing:
            - precision: Positive predictive value.
            - recall: True positive rate.
            - f1: Harmonic mean of precision and recall.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` contain different numbers of samples.

    Examples:
        >>> precision, recall, f1 = precision_recall_f1([1, 0, 1, 1], [1, 1, 1, 0])
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return precision, recall, float(f1)

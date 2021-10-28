"""
Score and performance functions.
"""
import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy classification score.

    Args:
        y_true (np.ndarray): correct target values.
        y_pred (np.ndarray): estimated targets as returned by a classifier.

    Returns:
        float: proportion of correctly classified samples.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes a confusion matrix.

    Args:
        y_true (np.ndarray): correct target values.
        y_pred (np.ndarray): estimated targets as returned by a classifier.

    Returns:
        np.ndarray: Confusion matrix whose i-th row and j-th column entry
        indicates the number of samples with true label being i-th class and
        predicted label being j-th class.
    """
    k = len(np.unique(y_true))  # Number of classes
    result = np.zeros((k, k), dtype=int)
    for i, j in zip(y_true, y_pred):
        result[int(i == 1), int(j == 1)] += 1
    return result


def get_proportions(y: np.ndarray) -> dict:
    """Gets proportions of the different values as a dictionary.

    Args:
        y (np.ndarray): vector.

    Returns:
        dict: proportions of values.
    """
    uniques, counts = np.unique(y, return_counts=True)
    return dict(zip(uniques, counts / y.size))


def get_proportion_empty(x: np.ndarray, value: int = -999) -> float:
    """Gets proportion of empty cells in each features.

    Args:
        x (np.ndarray): matrix.
        value (int, optional): empty value. Defaults to -999.

    Returns:
        float: proportion of empty cells.
    """
    return np.count_nonzero(x == value, axis=0) / x.shape[0]

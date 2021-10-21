"""Loss function."""
import numpy as np

from helpers import sigmoid


def compute_least_squares_loss(y, tx, w):
    """Compute the loss for least squares."""
    n = len(y)
    e = y - (tx @ w)
    return (e @ e) / (2 * n)


def compute_logistic_regression_loss(y, tx, w):
    """Compute the loss for logistic regression (negative log likelihood)."""
    s = sigmoid(tx @ w)
    return -(y.T @ np.log(s) + (1 - y).T @ np.log(1 - s))

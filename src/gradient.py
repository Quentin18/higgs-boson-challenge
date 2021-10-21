"""Gradient functions."""
import numpy as np

from helpers import sigmoid


def compute_least_squares_gradient(y, tx, w):
    """Compute the least squares gradient."""
    n = len(y)
    e = y - (tx @ w)
    return -(tx.T @ e) / n


def compute_logistic_regression_gradient(y, tx, w):
    """Compute the gradient of loss for logistic regression."""
    s = sigmoid(tx @ w)
    return tx.T @ (s - y)


def compute_logistic_regression_hessian(tx, w):
    """Compute the Hessian of loss for logistic regression."""
    s = sigmoid(tx @ w)
    s = np.diag(s.T[0])
    r = np.multiply(s, (1 - s))
    return tx.T @ r @ tx

"""
Gradient functions.
"""
import numpy as np

from helpers import sigmoid


def least_squares_gradient(y: np.ndarray, tx: np.ndarray,
                           w: np.ndarray) -> np.ndarray:
    """Compute the least squares gradient."""
    n = len(y)
    e = y - (tx @ w)
    return -(tx.T @ e) / n


def logistic_regression_gradient(y: np.ndarray, tx: np.ndarray,
                                 w: np.ndarray) -> np.ndarray:
    """Compute the gradient of loss for logistic regression."""
    s = sigmoid(tx @ w)
    return tx.T @ (s - y)


def reg_logistic_regression_gradient(y: np.ndarray, tx: np.ndarray,
                                     w: np.ndarray,
                                     lambda_: float) -> np.ndarray:
    """Compute the gradient of loss for regularized logistic regression."""
    w_norm = np.linalg.norm(w, 2)
    return logistic_regression_gradient(y, tx, w) + 2 * lambda_ * w_norm


def logistic_regression_hessian(tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the Hessian of loss for logistic regression."""
    s = sigmoid(tx @ w)
    s = np.diag(s.T[0])
    r = np.multiply(s, (1 - s))
    return tx.T @ r @ tx

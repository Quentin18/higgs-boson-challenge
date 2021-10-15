"""Machine Learning implementations."""
import numpy as np

from gradient import compute_least_squares_gradient
from helpers import batch_iter
from loss import compute_loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_least_squares_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_least_squares_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    a = (tx.T @ tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

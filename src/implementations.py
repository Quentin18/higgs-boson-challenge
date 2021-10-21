"""Machine Learning implementations."""
import numpy as np

from gradient import (compute_least_squares_gradient,
                      compute_logistic_regression_gradient)
from helpers import batch_iter
from loss import compute_least_squares_loss, compute_logistic_regression_loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = compute_least_squares_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_least_squares_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_least_squares_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
    loss = compute_least_squares_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_least_squares_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    a = (tx.T @ tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_least_squares_loss(y, tx, w)
    return w, loss


def logistic_regression_GD(y, tx, initial_w, max_iters, gamma,
                           threshold=1e-8, info=False):
    """Logistic regression using gradient descent or SGD."""
    losses = []
    w = initial_w
    for iter in range(max_iters):
        # Compute loss and gradient.
        loss = compute_logistic_regression_loss(y, tx, w)
        grad = compute_logistic_regression_gradient(y, tx, w)

        # Update w.
        w = w - gamma * grad

        # Display information.
        if iter % 100 == 0 and info:
            print(f'Iteration = {iter}, loss = {loss}')

        # Convergence criterion.
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,
                            threshold=1e-8, info=False):
    """Regularized logistic regression using gradient descent or SGD."""
    losses = []
    w = initial_w
    for iter in range(max_iters):
        # Compute loss and gradient.
        w_norm = np.linalg.norm(w, 2)
        loss = compute_logistic_regression_loss(y, tx, w) + lambda_ * w_norm**2
        grad = compute_logistic_regression_gradient(y, tx, w) \
            + 2 * lambda_ * w_norm

        # Update w.
        w = w - gamma * grad

        # Display information.
        if iter % 100 == 0 and info:
            print(f'Iteration = {iter}, loss = {loss}')

        # Convergence criterion.
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss

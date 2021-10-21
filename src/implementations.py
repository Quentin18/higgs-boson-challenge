"""Machine Learning implementations."""
import numpy as np

from gradient import (least_squares_gradient, logistic_regression_gradient,
                      reg_logistic_regression_gradient)
from helpers import batch_iter
from loss import (least_squares_loss, logistic_regression_loss,
                  reg_logistic_regression_loss)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        grad = least_squares_gradient(y, tx, w)
        w = w - gamma * grad
    loss = least_squares_loss(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Linear regression using stochastic gradient descent."""
    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = least_squares_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
    loss = least_squares_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = least_squares_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    a = (tx.T @ tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = least_squares_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma,
                        threshold=1e-8, info=False, info_step=100,
                        sgd=False, batch_size=1):
    """Logistic regression using gradient descent or SGD."""
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.append(initial_w, 1)
    for iter in range(max_iters):
        # Stochastic gradient descent
        if sgd:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = logistic_regression_gradient(
                    minibatch_y, minibatch_tx, w)
                w = w - gamma * grad
        # Gradient descent
        else:
            grad = logistic_regression_gradient(y, tx, w)
            w = w - gamma * grad

        # Compute loss
        loss = logistic_regression_loss(y, tx, w)

        # Display information
        if info and iter % info_step == 0:
            print(f'Iteration = {iter}, loss = {loss}')

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w[:-1], loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,
                            threshold=1e-8, info=False, info_step=100,
                            sgd=False, batch_size=1):
    """Regularized logistic regression using gradient descent or SGD."""
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.append(initial_w, 1)
    for iter in range(max_iters):
        # Stochastic gradient descent
        if sgd:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = reg_logistic_regression_gradient(
                    minibatch_y, minibatch_tx, w, lambda_)
                w = w - gamma * grad
        # Gradient descent
        else:
            grad = reg_logistic_regression_gradient(y, tx, w, lambda_)
            w = w - gamma * grad

        # Compute loss
        loss = reg_logistic_regression_loss(y, tx, w, lambda_)

        # Display information
        if info and iter % info_step == 0:
            print(f'Iteration = {iter}, loss = {loss}')

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w[:-1], loss

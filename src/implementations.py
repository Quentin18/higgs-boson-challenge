"""
Machine Learning algorithms implementations.
"""
import time

import numpy as np

from gradient import (least_squares_gradient, logistic_regression_gradient,
                      reg_logistic_regression_gradient)
from helpers import batch_iter
from loss import (least_squares_loss, logistic_regression_loss,
                  reg_logistic_regression_loss)
from print_utils import print_end, print_loss, print_progress, print_start


def least_squares_GD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                     max_iters: int, gamma: float, verbose: int = 1,
                     info_step: int = 100) -> tuple:
    """Linear regression using gradient descent.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        initial_w (np.ndarray): initial weights.
        max_iters (int): maximum iteration.
        gamma (float): stepsize.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        info_step (int, optional): step to print informations. Defaults to 100.

    Returns:
        tuple: w, loss.
    """
    proc_name = 'Gradient descent (least squares)'
    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    w = initial_w
    for iter in range(max_iters):
        # Compute gradient
        grad = least_squares_gradient(y, tx, w)

        # Update weights
        w = w - gamma * grad

        # Compute loss
        loss = least_squares_loss(y, tx, w)

        # Display information
        if verbose >= 2 and iter % info_step == 0:
            print_progress(iter, max_iters, loss)

    if verbose >= 1:
        print_end(proc_name, time.time() - t_start)

    return w, loss


def least_squares_SGD(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                      max_iters: int, gamma: float, batch_size: int = 1,
                      verbose: int = 1, info_step: int = 100) -> tuple:
    """Linear regression using stochastic gradient descent.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        initial_w (np.ndarray): initial weights.
        max_iters (int): maximum iteration.
        gamma (float): stepsize.
        batch_size (int, optional): batch size. Defaults to 1.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        info_step (int, optional): step to print informations. Defaults to 100.

    Returns:
        tuple: w, loss.
    """
    proc_name = 'Stochastic gradient descent (least squares)'
    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute gradient
            grad = least_squares_gradient(minibatch_y, minibatch_tx, w)

            # Update weights
            w = w - gamma * grad

        # Compute loss
        loss = least_squares_loss(y, tx, w)

        # Display information
        if verbose >= 2 and iter % info_step == 0:
            print_progress(iter, max_iters, loss)

    if verbose >= 1:
        print_end(proc_name, time.time() - t_start)

    return w, loss


def least_squares(y: np.ndarray, tx: np.ndarray,
                  verbose: bool = False) -> tuple:
    """Least squares regression using normal equations.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        verbose (bool, optional): True to print loss. Defaults to False.

    Returns:
        tuple: w, loss.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = least_squares_loss(y, tx, w)

    if verbose:
        print_loss(loss)

    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float,
                     verbose: bool = False) -> tuple:
    """Ridge regression using normal equations.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        lambda_ (float): regularization parameter.
        verbose (bool, optional): True to print loss. Defaults to False.

    Returns:
        tuple: w, loss.
    """
    n, d = tx.shape
    a = (tx.T @ tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = least_squares_loss(y, tx, w)

    if verbose:
        print_loss(loss)

    return w, loss


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                        max_iters: int, gamma: float,
                        threshold: float = 1e-8, verbose: int = 1,
                        info_step: int = 100, sgd: bool = False,
                        agd: bool = False, batch_size: int = 1) -> tuple:
    """Logistic regression using gradient descent or SGD.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        initial_w (np.ndarray): initial weights.
        max_iters (int): maximum iteration.
        gamma (float): stepsize.
        threshold (float, optional): threshold to stop. Defaults to 1e-8.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        info_step (int, optional): step to print informations. Defaults to 100.
        sgd (bool, optional): True to use SGD. Defaults to False.
        agd (bool, optional): True to use AGD. Defaults to False.
        batch_size (int, optional): batch size. Defaults to 1.

    Returns:
        tuple: w, loss.
    """
    proc_name = 'Logistic regression'
    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 0)
    if agd:
        # Init variables for agd
        z, t = np.copy(w), 1

    for iter in range(max_iters):
        # Stochastic gradient descent
        if sgd:
            for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
                grad = logistic_regression_gradient(
                    minibatch_y, minibatch_tx, w)
                w = w - gamma * grad

        # Accelerated gradient descent
        elif agd:
            grad = logistic_regression_gradient(y, tx, z)
            w_next = z - gamma * grad
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
            z = w_next + (w_next - w) * (t - 1) / t_next
            w = w_next
            t = t_next

        # Gradient descent
        else:
            grad = logistic_regression_gradient(y, tx, w)
            w = w - gamma * grad

        # Compute loss
        loss = logistic_regression_loss(y, tx, w)

        # Display information
        if verbose >= 2 and iter % info_step == 0:
            print_progress(iter, max_iters, loss)

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    if verbose >= 1:
        print_end(proc_name, time.time() - t_start)

    return w, loss


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float,
                            initial_w: np.ndarray, max_iters: int,
                            gamma: float, threshold: float = 1e-8,
                            verbose: int = 1, info_step: int = 100,
                            sgd: bool = False, batch_size: int = 1) -> tuple:
    """Regularized logistic regression using gradient descent or SGD.

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        lambda_ (float): regularization parameter.
        initial_w (np.ndarray): initial weights.
        max_iters (int): maximum iteration.
        gamma (float): stepsize.
        threshold (float, optional): threshold to stop. Defaults to 1e-8.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        info_step (int, optional): step to print informations. Defaults to 100.
        sgd (bool, optional): True to use SGD. Defaults to False.
        batch_size (int, optional): batch size. Defaults to 1.

    Returns:
        tuple: w, loss.
    """
    proc_name = 'Regularized logistic regression'
    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 0)

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
        if verbose >= 2 and iter % info_step == 0:
            print_progress(iter, max_iters, loss)

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    if verbose >= 1:
        print_end(proc_name, time.time() - t_start)

    return w, loss

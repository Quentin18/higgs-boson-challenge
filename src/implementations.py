"""
Machine Learning algorithms implementations.
"""
import numpy as np
import matplotlib.pyplot as plt

from gradient import (least_squares_gradient, logistic_regression_gradient,
                      reg_logistic_regression_gradient)
from helpers import batch_iter
from loss import (least_squares_loss, logistic_regression_loss,
                  reg_logistic_regression_loss)
from cross_validation import build_k_indices,cross_validation_lambda


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

def ridge_regression_cross_validation(y: np.ndarray, x: np.ndarray, lambdas: np.ndarray,k_fold: int=4,plot: bool=False):
    """Ridge regression using normal equations and k-fold cross validation on the lambda parameter."""
    # split data in k fold
    k_indices = build_k_indices(y, k_fold)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    # cross validation
    for lambda_ in lambdas:
        loss_tr_tmp = []
        loss_te_tmp = []
        for k in range(k_fold):
            loss_tr_k, loss_te_k,_ = cross_validation_lambda(y, x, k_indices, k, lambda_)
            loss_tr_tmp.append(loss_tr_k)
            loss_te_tmp.append(loss_te_k)
        loss_tr.append(np.mean(loss_tr_tmp))
        loss_te.append(np.mean(loss_te_tmp))

    if plot:
        plt.plot(lambdas,loss_tr,color='blue')
        plt.plot(lambdas,loss_te,color='red')
        plt.legend(['training loss','test loss'])

    best_lambda=lambdas[np.argmin(loss_te)]
    print(best_lambda)
    w,loss=ridge_regression(y,x,best_lambda)
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma,
                        threshold=1e-8, info=False, info_step=100,
                        sgd=False, agd=False, batch_size=1):
    """Logistic regression using gradient descent or SGD."""
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
        if info and iter % info_step == 0:
            print(f'Iter: {iter:05}/{max_iters} - Loss: {loss:.2f}')

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma,
                            threshold=1e-8, info=False, info_step=100,
                            sgd=False, batch_size=1):
    """Regularized logistic regression using gradient descent or SGD."""
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
        if info and iter % info_step == 0:
            print(f'Iter: {iter:05}/{max_iters} - Loss: {loss:.2f}')

        # Convergence criterion
        losses.append(loss)
        if iter > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def adagrad(y, tx, lambda_, initial_w, max_iters, info=False, info_step=100):
    """Adaptative gradient method for regularized logistic regression."""
    q = 0
    gamma = 1
    delta = 1e-5
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 0)
    for iter in range(max_iters):
        # Compute gradient.
        grad = reg_logistic_regression_gradient(y, tx, w, lambda_)

        # Update the next iteration.
        q += np.linalg.norm(grad)**2
        h = (np.sqrt(q) + delta) * np.eye(grad.size)
        w = w - gamma * np.linalg.solve(h, grad)

        # Compute loss
        loss = reg_logistic_regression_loss(y, tx, w, lambda_)

        # Display information
        if info and iter % info_step == 0:
            print(f'Iter: {iter:05}/{max_iters} - Loss: {loss:.2f}')

    return w, loss

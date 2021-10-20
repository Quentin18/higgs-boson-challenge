"""Machine Learning implementations."""
import numpy as np

from gradient import compute_least_squares_gradient
from helpers import batch_iter
from loss import compute_loss, compute_logistic_reg_loss
from gradient import compute_logistic_reg_gradient

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

def logistic_regression_GD(y, x, initial_w, gamma, max_iter=10000, threshold=1e-8, info=False):
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    # initialize w0 = zeros
    w = initial_w
    for iter in range(max_iter):
        # get loss and update w.
        loss=compute_logistic_reg_loss(y, tx, w)
        grad=compute_logistic_reg_gradient(y, tx, w)
        w=w-gamma*grad
        # disp information if info=True
        if iter % 100 == 0 and info:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    loss=compute_logistic_reg_loss(y, tx, w)    
    return w, loss

def reg_logistic_regression(y, x, initial_w, lambda_, gamma, max_iter=10000, threshold=1e-8, info=False):
    losses = []
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w
    for iter in range(max_iter):
        # get loss and update w.
        loss=compute_logistic_reg_loss(y,tx,w)+lambda_* np.power(np.linalg.norm(w),2)
        grad=compute_logistic_reg_gradient(y,tx,w)+2*lambda_*w
        w=w-gamma*grad
        # disp information if info=True
        if iter % 100 == 0 and info:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    loss=compute_logistic_reg_loss(y,tx,w)+lambda_* np.power(np.linalg.norm(w),2)
    return w, loss
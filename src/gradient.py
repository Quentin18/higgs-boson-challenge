"""Gradient functions."""
import numpy as np
from helpers import sigmoid

def compute_least_squares_gradient(y, tx, w):
    """Compute the least squares gradient."""
    n = len(y)
    e = y - (tx @ w)
    return -(tx.T @ e) / n

def compute_logistic_reg_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx.dot(w))-y)

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    # calculate Hessian: TODO
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    S=np.multiply(pred,1-pred)
    return tx.T.dot(S).dot(tx)

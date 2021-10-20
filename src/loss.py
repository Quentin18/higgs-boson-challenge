"""Loss function."""

import numpy as np
from helpers import sigmoid

def compute_loss(y, tx, w):
    """Calculate the loss."""
    n = len(y)
    e = y - (tx @ w)
    return (e @ e) / (2 * n)

def compute_logistic_reg_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

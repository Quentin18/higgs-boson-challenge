import numpy as np
import implementations
from loss import least_squares_loss


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_lambda(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice, :]
    x_tr = x[tr_indice, :]
    # form data with polynomial degree
    # ridge regression
    w, loss_tr = implementations.ridge_regression(y_tr, x_tr, lambda_)
    # calculate the loss for train and test data
    loss_te = least_squares_loss(y_te, x_te, w)
    return loss_tr, loss_te, w

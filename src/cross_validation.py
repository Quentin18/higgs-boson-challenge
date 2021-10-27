"""
Cross validation functions.
"""
import time

import numpy as np

import implementations
import loss
from clean_data import standardize
from helpers import predict_labels
from metrics import accuracy_score


def build_k_indices(y: np.ndarray, k_fold: int) -> np.ndarray:
    """Builds k indices for k-fold.

    Args:
        y (np.ndarray): input data.
        k_fold (int): number of folds.

    Returns:
        np.ndarray: array of indices.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """Builds polynomial basis functions for input data x,
    for d = 0 up to d = degree.

    Args:
        x (np.ndarray): input data.
        degree (int): degree of polynomial.

    Returns:
        np.ndarray: polynomial basis functions.
    """
    if x.shape[0] == 1:
        return np.vander(x, degree + 1, increasing=True)
    x_expanded = np.copy(x)
    for d in range(2, degree + 1):
        x_expanded = np.concatenate((x_expanded, x**d), axis=1)
    return x_expanded


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
    loss_te = loss.least_squares_loss(y_te, x_te, w)
    return loss_tr, loss_te, w


def cross_validation_poly(y: np.ndarray, x: np.ndarray, gamma: float,
                          k_indices: int, k: int, degree: int,
                          max_iters: int = 1000) -> tuple:
    """Performs an iteration of cross validation for degree of polynomials.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): output desired values.
        gamma (float): direction.
        k_indices (int): indices of fold k.
        k (int): fold number.
        degree (int): degree of polynomials.
        max_iters (int, optional): maximum number of iterations.
        Defaults to 1000.

    Returns:
        tuple: acc_tr, acc_te.
    """
    threshold = 1e-8

    # Get k'th subgroup in test, others in train
    ind_te = k_indices[k]
    ind_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    ind_tr = ind_tr.reshape(-1)
    x_tr, x_te, y_tr, y_te = x[ind_tr], x[ind_te], y[ind_tr], y[ind_te]

    # Form data with polynomial degree
    x_tr_poly = build_poly(x_tr, degree)
    x_te_poly = build_poly(x_te, degree)

    # Standerdize data
    x_tr_poly = standardize(x_tr_poly)
    x_te_poly = standardize(x_te_poly)

    # Logistic regression
    initial_w = np.zeros((x_tr_poly.shape[1], 1))
    w, _ = implementations.logistic_regression(
        y_tr, x_tr_poly, initial_w, max_iters, gamma, threshold,
        agd=True, info=False)

    # Predict labels
    x_tr_poly = np.c_[np.ones((y_tr.shape[0], 1)), x_tr_poly]
    x_te_poly = np.c_[np.ones((y_te.shape[0], 1)), x_te_poly]
    y_pred_tr = predict_labels(w, x_tr_poly, label_b_in=0, label_b_out=0,
                               use_sigmoid=True)
    y_pred_te = predict_labels(w, x_te_poly, label_b_in=0, label_b_out=0,
                               use_sigmoid=True)

    # Calculate accuracy
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_te = accuracy_score(y_te, y_pred_te)

    return acc_tr, acc_te


def get_best_degree(y: np.ndarray, x: np.ndarray, gamma: float,
                    degrees: list, k_fold: int = 4, max_iters: int = 1000,
                    verbose: bool = True) -> int:
    """Returns the best degree determined by cross validation.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): output desired values.
        gamma (float): direction.
        degrees (list): degrees to test.
        k_fold (int, optional): number of folds. Defaults to 4.
        max_iters (int, optional): maximum number of iterations.
        Defaults to 1000.
        verbose (bool, optional): True fo show infos. Defaults to True.

    Returns:
        int: best degree.
    """
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # Define lists to store accuracy of training data and test data
    acc_tr, acc_te = list(), list()

    if verbose:
        print('[Start] Cross validation')

    t_start = time.time()

    # Cross validation
    for degree in degrees:
        acc_tr_k, acc_te_k = list(), list()
        for k in range(k_fold):
            # Calculate the accuracy for train and test data
            a_tr, a_te = cross_validation_poly(y, x, gamma, k_indices, k,
                                               degree, max_iters)

            # Add accuracies to lists
            acc_tr_k.append(a_tr)
            acc_te_k.append(a_te)

        # Calculate means and add to lists
        acc_tr.append(np.mean(acc_tr_k))
        acc_te.append(np.mean(acc_te_k))

        if verbose:
            print(f'[CP] Degree = {degree}, Accuracy = {acc_te[-1]:.3f}')

    # Return best degree
    best_degree = degrees[np.argmin(acc_te)]

    if verbose:
        print(
            f'[End] Cross validation (time: {time.time() - t_start: .2f} s.)')
        print('[Results] Best degree:', best_degree)

    return best_degree

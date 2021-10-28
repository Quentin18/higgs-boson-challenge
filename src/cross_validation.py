"""
Cross validation functions.
"""
import time
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np

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


def cross_validation_iter(y: np.ndarray, x: np.ndarray, optimizer: Callable,
                          k_indices: int, k: int, param: Union[int, float],
                          param_name: str, **kwargs) -> tuple:
    """Performs an iteration of cross validation and returns accuracies.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): output desired values.
        optimizer (Callable): function of "implementations" to use.
        k_indices (int): indices of fold k.
        k (int): fold number.
        param (int or float): value of parameter.
        param_name (str): name of the parameter to find ('degree' or 'lambda').

    Returns:
        tuple: acc_tr, acc_te.
    """
    # Get k'th subgroup in test, others in train
    ind_te = k_indices[k]
    ind_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    ind_tr = ind_tr.reshape(-1)
    x_tr, x_te, y_tr, y_te = x[ind_tr], x[ind_te], y[ind_tr], y[ind_te]

    # Form data with polynomial degree
    if param_name == 'degree':
        x_tr = build_poly(x_tr, param)
        x_te = build_poly(x_te, param)
    else:
        kwargs[param_name] = param

    # Standerdize data
    x_tr = standardize(x_tr)
    x_te = standardize(x_te)

    # Run optimization
    w, _ = optimizer(y_tr, x_tr, **kwargs)
    # initial_w = np.zeros((x_tr_poly.shape[1], 1))

    # Predict labels
    # x_tr_poly = np.c_[np.ones((y_tr.shape[0], 1)), x_tr_poly]
    # x_te_poly = np.c_[np.ones((y_te.shape[0], 1)), x_te_poly]
    y_pred_tr = predict_labels(w, x_tr, label_b_in=0, label_b_out=0,
                               use_sigmoid=False)
    y_pred_te = predict_labels(w, x_te, label_b_in=0, label_b_out=0,
                               use_sigmoid=False)

    # Calculate accuracy
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_te = accuracy_score(y_te, y_pred_te)

    return acc_tr, acc_te


def get_best_param(y: np.ndarray, x: np.ndarray, optimizer: Callable,
                   param_name: str, param_list: list, k_fold: int = 4,
                   verbose: bool = True, plot: bool = False,
                   title: str = 'Cross validation results',
                   **kwargs) -> Union[int, float]:
    """Returns the best parameter determined by cross validation.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        optimizer (Callable): function of "implementations" to use.
        param_name (str): name of the parameter to find ('degree' or 'lambda').
        param_list (list): list of parameters to test.
        k_fold (int, optional): number of folds. Defaults to 4.
        verbose (bool, optional): True fo show infos. Defaults to True.
        plot (bool, optional): True to plot results of cross validation.
        Defaults to False.
        title (str, optional): title of the plot.
        Defaults to 'Cross validation results'.

    Returns:
        int: best parameter.
    """
    # Split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # Define lists to store accuracy of training data and test data
    acc_tr, acc_te = list(), list()

    if verbose:
        print('[Start] Cross validation')

    t_start = time.time()

    # Cross validation
    for param in param_list:
        acc_tr_k, acc_te_k = list(), list()
        for k in range(k_fold):
            # Calculate the accuracy for train and test data
            a_tr, a_te = cross_validation_iter(y, x, optimizer, k_indices, k,
                                               param, param_name, **kwargs)

            # Add accuracies to lists
            acc_tr_k.append(a_tr)
            acc_te_k.append(a_te)

        # Calculate means and add to lists
        acc_tr.append(np.mean(acc_tr_k))
        acc_te.append(np.mean(acc_te_k))

        if verbose:
            print(f'[CP] {param_name.capitalize()} = {param}, '
                  f'Accuracy = {acc_te[-1]:.3f}')

    # Return best param
    best_param = param_list[np.argmax(acc_te)]

    if verbose:
        print(
            f'[End] Cross validation (time: {time.time() - t_start: .2f} s.)')
        print(f'[Results] Best {param_name}: {best_param}')

    if plot:
        plt.plot(param_list, acc_tr, marker='*', label='Train accuracy')
        plt.plot(param_list, acc_te, marker='*', label='Test accuracy')
        plt.scatter(x=best_param, y=np.max(acc_te), c='r',
                    label=f'Best {param_name}')
        plt.title(title)
        plt.legend()

    return best_param

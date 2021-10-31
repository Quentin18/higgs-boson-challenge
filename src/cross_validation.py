"""
Cross validation functions.
"""
import time
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np

from helpers import predict_labels
from metrics import accuracy_score
from print_utils import print_end, print_start


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
                          param_name: str, logi=False, **kwargs) -> tuple:
    """Performs an iteration of cross validation and returns accuracies.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
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
        # Give parameter to the optimizer
    else:
        kwargs[param_name] = param

    if logi:
        kwargs["initial_w"] = np.zeros((x_tr.shape[1], 1))

    # Run optimization
    w, _ = optimizer(y_tr, x_tr, **kwargs)

    if logi:
        print("eee")
        x_tr = np.c_[np.ones((y_tr.shape[0], 1)), x_tr]
        x_te = np.c_[np.ones((y_te.shape[0], 1)), x_te]
        y_pred_tr = predict_labels(w, x_tr, label_b=0, use_sigmoid=True)
        y_pred_te = predict_labels(w, x_te, label_b=0, use_sigmoid=True)
    else:
        y_pred_tr = predict_labels(w, x_tr)
        y_pred_te = predict_labels(w, x_te)

    # Calculate accuracy
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_te = accuracy_score(y_te, y_pred_te)

    return acc_tr, acc_te


def get_best_param(y: np.ndarray, x: np.ndarray, optimizer: Callable,
                   param_name: str, param_list: list, k_fold: int = 4,
                   verbose: int = 1, plot: bool = False,
                   title: str = 'Cross validation results', logi: bool = False,
                   **kwargs) -> Union[int, float]:
    """Returns the best parameter determined by cross validation.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        optimizer (Callable): function of "implementations" to use.
        param_name (str): name of the parameter to find ('degree' or 'lambda').
        param_list (list): list of parameters to test.
        k_fold (int, optional): number of folds. Defaults to 4.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        plot (bool, optional): True to plot results of cross validation.
        Defaults to False.
        title (str, optional): title of the plot.
        Defaults to 'Cross validation results'.

    Returns:
        int or float: best parameter.
    """
    proc_name = 'Cross validation'

    # Split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # Define lists to store accuracy of training data and test data
    acc_tr, acc_te = list(), list()

    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    # Cross validation
    for param in param_list:
        acc_tr_k, acc_te_k = list(), list()
        for k in range(k_fold):
            # Calculate the accuracy for train and test data
            a_tr, a_te = cross_validation_iter(y, x, optimizer, k_indices, k,
                                               param, param_name, logi=logi,
                                               **kwargs)

            # Add accuracies to lists
            acc_tr_k.append(a_tr)
            acc_te_k.append(a_te)

        # Calculate means and add to lists
        acc_tr.append(np.mean(acc_tr_k))
        acc_te.append(np.mean(acc_te_k))

        if verbose >= 2:
            print(f'[CP] {param_name.capitalize()} = {param}, '
                  f'Accuracy = {acc_te[-1]:.3f}')

    # Get best param
    best_param = param_list[np.argmax(acc_te)]
    best_accuracy = np.max(acc_te)

    if verbose >= 1:
        print('[Results]')
        print(f'- Best {param_name}: {best_param}')
        print(f'- Best accuracy: {best_accuracy:.3f}')
        print_end(proc_name, time.time() - t_start)

    if plot:
        plt.plot(param_list, acc_tr, marker='*', label='Train accuracy')
        plt.plot(param_list, acc_te, marker='*', label='Test accuracy')
        plt.scatter(x=best_param, y=np.max(acc_te), c='r',
                    label=f'Best {param_name}')
        plt.title(title)
        plt.legend()

    return best_param


def get_best_lambda_degree(y: np.ndarray, x: np.ndarray, optimizer: Callable,
                           lambdas: list, degrees: list, k_fold: int = 4,
                           verbose: int = 1, plot: bool = False,
                           title: str = 'Cross validation results',
                           **kwargs) -> list:
    """Returns the best tuple (lambda, degree) determined by cross validation.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        optimizer (Callable): function of "implementations" to use.
        lambdas (list): list of lambdas to test.
        degrees (list): list of degrees to test.
        k_fold (int, optional): number of folds. Defaults to 4.
        verbose (int, optional): verbose level (0, 1 or 2). Defaults to 1.
        plot (bool, optional): True to plot results of cross validation.
        Defaults to False.
        title (str, optional): title of the plot.
        Defaults to 'Cross validation results'.

    Returns:
        list: best parameters.
    """
    proc_name = 'Cross validation'

    # Split data in k fold
    k_indices = build_k_indices(y, k_fold)

    # Define lists to store accuracy of training data and test data
    # and tuples of parameters (lambda, degree)
    acc_tr, acc_te, params_list = list(), list(), list()

    if verbose >= 1:
        print_start(proc_name)

    t_start = time.time()

    mat = np.zeros((len(lambdas), len(degrees)))

    # Cross validation
    for i, lambda_ in enumerate(lambdas):
        for j, degree in enumerate(degrees):
            acc_tr_k, acc_te_k = list(), list()
            for k in range(k_fold):
                # Calculate the accuracy for train and test data
                kwargs['lambda_'] = lambda_
                a_tr, a_te = cross_validation_iter(
                    y, x, optimizer, k_indices, k, degree, 'degree', **kwargs)

                # Add accuracies to lists
                acc_tr_k.append(a_tr)
                acc_te_k.append(a_te)

            # Calculate means and add to lists
            acc_tr.append(np.mean(acc_tr_k))
            acc_te.append(np.mean(acc_te_k))
            params_list.append((lambda_, degree))
            mat[i][j] = np.mean(acc_te_k)

            if verbose >= 2:
                print(f'[CP] (lambda, degree) = {params_list[-1]}, '
                      f'Accuracy = {acc_te[-1]:.3f}')

    # Get best param
    best_params = params_list[np.argmax(acc_te)]
    best_accuracy = np.max(acc_te)

    if verbose >= 1:
        print('[Results]')
        print(f'- Best (lambda, degree): {best_params}')
        print(f'- Best accuracy: {best_accuracy:.3f}')
        print_end(proc_name, time.time() - t_start)

    if plot:
        plt.imshow(mat)

        plt.xticks(np.arange(len(degrees)), degrees)
        plt.yticks(np.arange(len(lambdas)),
                   [f'{lambda_:.5f}' for lambda_ in lambdas])

        plt.xlabel('Degree')
        plt.ylabel('Lambda')
        plt.title(title)

    return best_params

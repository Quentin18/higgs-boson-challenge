"""
Split functions to handle data.
"""
import numpy as np

from clean_data import clean_data_by_jet


def split_train_test(y: np.ndarray, x: np.ndarray, ratio: float = 0.8,
                     seed: int = 1) -> tuple:
    """Splits the data into random train and test subsets.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): output desired values.
        ratio (float, optional): split ratio. Defaults to 0.8.
        seed (int, optional): seed for random generator. Defaults to 1.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """
    # Set seed
    np.random.seed(seed)

    # Get split index
    n = x.shape[0]
    split_idx = int(n * ratio)

    # Generate indices
    indices = np.random.permutation(n)
    idx_train, idx_test = indices[:split_idx], indices[split_idx:]

    # Split data
    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    return x_train, x_test, y_train, y_test


def split_by_jet(y: np.ndarray, x: np.ndarray, ids: np.ndarray,
                 jet_col_num: int = 22, max_jet: int = 3,
                 clean: bool = True) -> tuple:
    """Splits the dataset by jet.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        ids (np.ndarray): ids of rows.
        jet_col_num (int, optional): index of jet column. Defaults to 22.
        max_jet (int, optional): maximum number of jet. Defaults to 3.
        clean (bool, optional): True to clean data. Defaults to True.
    Returns:
        tuple: y_by_jet, x_by_jet, ids_by_jet.
    """
    jet_col = x[:, jet_col_num]
    y_by_jet, x_by_jet, ids_by_jet = list(), list(), list()
    for i in range(max_jet + 1):
        rows_indices = np.where(jet_col == i)   # select rows indices
        rows = np.squeeze(x[rows_indices, :])   # select rows
        rows_without_jet = np.delete(rows, jet_col_num, axis=1)
        x_by_jet.append(rows_without_jet)       # add rows to list
        y_by_jet.append(y[rows_indices])
        ids_by_jet.append(ids[rows_indices])

    if clean:
        clean_data_by_jet(x_by_jet)

    return y_by_jet, x_by_jet, ids_by_jet


def split_by_label(y: np.ndarray, x: np.ndarray, label_b: int = -1, plot: bool=False) -> tuple:
    """Split the dataset whith respect to label.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        label_b (int, optional): value for label b (0 or -1). Defaults to -1.

    Returns:
        tuple: x_label_s, x_label_b
    """
    ind_label_s = np.where(y == 1)
    ind_label_b = np.where(y == label_b)
    if plot:
        x_label_s = x[ind_label_s]
        x_label_b = x[ind_label_b]
    else:
        x_label_s = x[ind_label_s, :]
        x_label_b = x[ind_label_b, :]
    return np.squeeze(x_label_s), np.squeeze(x_label_b)

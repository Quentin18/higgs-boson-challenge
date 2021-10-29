"""
Split functions to handle data.
"""
import numpy as np

# Column of JET feature
JET_COL_NUM = 22

# Number of subsets
NB_SUBSETS = 3


def split_train_test(y: np.ndarray, x: np.ndarray, ratio: float = 0.8,
                     seed: int = 1) -> tuple:
    """Splits the data into random train and test subsets.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        ratio (float, optional): split ratio. Defaults to 0.8.
        seed (int, optional): seed for random generator. Defaults to 1.

    Returns:
        tuple: x_train, x_test, y_train, y_test.
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


def split_by_jet(y: np.ndarray, x: np.ndarray,
                 ids: np.ndarray = None) -> tuple:
    """Splits the dataset by jet.

    It returns 3 subsets:
    - JET == 0
    - JET == 1
    - JET >= 2

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        ids (np.ndarray, optional): ids of rows. Defaults to None.
    Returns:
        tuple: y_by_jet, x_by_jet, ids_by_jet (empty if ids is None).
    """
    jet_col = x[:, JET_COL_NUM]
    y_by_jet, x_by_jet, ids_by_jet = list(), list(), list()
    for i in range(NB_SUBSETS):
        # Select rows indices
        if i == 2:
            rows_indices = np.where(jet_col >= i)
        else:
            rows_indices = np.where(jet_col == i)

        # Select rows
        rows = np.squeeze(x[rows_indices, :])

        # Delete JET column
        rows_without_jet = np.delete(rows, JET_COL_NUM, axis=1)

        # Add subset to list
        x_by_jet.append(rows_without_jet)
        y_by_jet.append(y[rows_indices])
        if ids is not None:
            ids_by_jet.append(ids[rows_indices])

    return y_by_jet, x_by_jet, ids_by_jet


def split_by_label(y: np.ndarray, x: np.ndarray, label_b: int = -1,
                   plot: bool = False) -> tuple:
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

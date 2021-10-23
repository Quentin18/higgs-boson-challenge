"""
Helpers functions.
"""
import numpy as np


def batch_iter(y: np.ndarray, tx: np.ndarray, batch_size: int,
               num_batches: int = 1, shuffle: bool = True):
    """Minibatch iterator for a dataset.

    Data can be randomly shuffled to avoid ordering in the original data
    messing with the randomness of the minibatches.

    Example of use:
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    Args:
        y (np.ndarray): output desired values.
        tx (np.ndarray): input data.
        batch_size (int): batch size
        num_batches (int, optional): number of batches. Defaults to 1.
        shuffle (bool, optional): shuffle data. Defaults to True.

    Yields:
        tuple: mini-batches of batch_size matching elements from y and tx.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], \
                shuffled_tx[start_index:end_index]


def split_data(x: np.ndarray, y: np.ndarray, ratio: float,
               seed: int = 1) -> tuple:
    """Splits the data into random train and test subsets.

    Args:
        x (np.ndarray): input data.
        y (np.ndarray): output desired values.
        ratio (float): split ratio.
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


def sigmoid(t: float) -> float:
    """Applies the sigmoid function on t.

    Args:
        t (float): input.

    Returns:
        float: sigmoid function evaluated on t.
    """
    return 1 / (1 + np.exp(-t))


def replace_nan_to_zero(x: np.ndarray) -> np.ndarray:
    """Replaces the nan values by zero in x.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: matrix x with zeros instead of nan.
    """
    x[np.isnan(x)] = 0
    return x


def drop_columns_nan(x: np.ndarray, value: int = -999) -> np.ndarray:
    """Drops the columns of x containing a certain value.

    Args:
        x (np.ndarray): matrix.
        value (int, optional): value to remove. Defaults to -999.

    Returns:
        np.ndarray: matrix x witout columns containing value.
    """
    x = np.where(x == -999, np.nan, x)  # replace -999 to nan
    x = x[:, ~np.isnan(x).any(axis=0)]  # drop columns with nan
    return x

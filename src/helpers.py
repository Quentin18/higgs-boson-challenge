"""Helpers functions."""
import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Takes as input two iterables (here the output desired values 'y'
    and the input data 'tx')

    Outputs an iterator which gives mini-batches of `batch_size` matching
    elements from `y` and `tx`.

    Data can be randomly shuffled to avoid ordering in the original data
    messing with the randomness of the minibatches.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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


def split_data(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio."""
    np.random.seed(seed)
    n = x.shape[0]
    split_idx = int(n * ratio)
    indices = np.random.permutation(n)
    training_idx, test_idx = indices[:split_idx], indices[split_idx:]
    training_x, test_x = x[training_idx], x[test_idx]
    training_y, test_y = y[training_idx], y[test_idx]
    return training_x, test_x, training_y, test_y


def sigmoid(t):
    """Apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))

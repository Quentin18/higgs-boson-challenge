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
        batch_size (int): batch size.
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


def sigmoid(t: float) -> float:
    """Applies the sigmoid function on t.

    Args:
        t (float): input.

    Returns:
        float: sigmoid function evaluated on t.
    """
    return 1 / (1 + np.exp(-t))


def predict_labels(weights: np.ndarray, data: np.ndarray, label_b: int = -1,
                   use_sigmoid: bool = False) -> np.ndarray:
    """Generates class predictions given weights, and a test data matrix.

    Args:
        weights (np.ndarray): vector of weights.
        data (np.ndarray): matrix of test data.
        label_b (int, optional): label of "b" event (-1 or 0).
        Defaults to -1.
        use_sigmoid (bool, optional): True to use the sigmoid function to
        predict labels. Defaults to False.

    Returns:
        np.ndarray: class predictions.
    """
    if use_sigmoid:
        y_pred = sigmoid(np.dot(data, weights))
        threshold = 1 / 2
    else:
        y_pred = np.dot(data, weights)
        threshold = 0

    # Select class label
    y_pred[np.where(y_pred <= threshold)] = label_b
    y_pred[np.where(y_pred > threshold)] = 1

    # Convert result to array of int
    y_pred = y_pred.astype(int)

    return y_pred

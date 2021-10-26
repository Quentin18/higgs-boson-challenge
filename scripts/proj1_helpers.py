"""
Some helper functions for project 1.
"""
import csv

import numpy as np

from path import add_src_to_path

# Add src to path to import implementations
add_src_to_path()

from helpers import sigmoid


def load_csv_data(data_path: str, sub_sample: bool = False,
                  label_b: int = -1) -> tuple:
    """Loads data from a csv file.

    Args:
        data_path (str): path of data file.
        sub_sample (bool, optional): return a subsambple. Defaults to False.
        label_b (int, optional): label of "b" event (-1 or 0). Defaults to -1.

    Returns:
        tuple: y (class labels), tX (features) and ids (event ids).
    """
    y = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=str,
                      usecols=1)
    x = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1, 1) or (0, 1)
    yb = np.ones(len(y), dtype=int)
    yb[np.where(y == 'b')] = label_b

    # Creates a sub-sample if needed
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights: np.ndarray, data: np.ndarray,
                   label_b_in: int = -1, label_b_out: int = -1,
                   use_sigmoid: bool = False) -> np.ndarray:
    """Generates class predictions given weights, and a test data matrix.

    Args:
        weights (np.ndarray): vector of weights.
        data (np.ndarray): matrix of test data.
        label_b_in (int, optional): label of "b" event in input data
        (-1 or 0). Defaults to -1.
        label_b_out (int, optional): label of "b" event out input data
        (-1 or 0). Defaults to -1.
        use_sigmoid (bool, optional): True to use the sigmoid function to
        predict labels. Defaults to False.

    Returns:
        np.ndarray: class predictions.
    """
    border = (label_b_in + 1) / 2
    if use_sigmoid:
        y_pred = sigmoid(np.dot(data, weights))
    else:
        y_pred = np.dot(data, weights)

    # Select class label
    y_pred[np.where(y_pred <= border-0.1)] = label_b_out
    y_pred[np.where(y_pred > border-0.1)] = 1

    # Convert result to array of int
    y_pred = y_pred.astype(int)

    return y_pred


def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray, name: str):
    """Creates an output file in csv format for submission to
    Kaggle or AIcrowd.

    Args:
        ids (np.ndarray): event ids associated with each prediction.
        y_pred (np.ndarray): predicted class labels.
        name (str): name of csv output file to be created.
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

"""
Some helper functions to load data and create submission..
"""
import csv

import numpy as np


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

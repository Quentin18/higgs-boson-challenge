"""
Some helper functions to load data and create submission.
"""
import csv

import numpy as np

from path import create_out_dir, extract_archives


def load_csv_data(data_path: str, subsample: bool = False,
                  label_b: int = -1) -> tuple:
    """Loads data from a csv file.

    Args:
        data_path (str): path of data file.
        subsample (bool, optional): return a subsample. Defaults to False.
        label_b (int, optional): label of "b" event (-1 or 0). Defaults to -1.

    Returns:
        tuple: y (class labels), x (features) and ids (event ids).
    """
    # Extract archives if needed
    extract_archives()

    # Load data
    y = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=str,
                      usecols=1)
    x = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1, 1) or (0, 1)
    yb = np.ones(len(y), dtype=int)
    yb[np.where(y == 'b')] = label_b

    # Creates a subsample if needed
    if subsample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray,
                          filename: str) -> None:
    """Creates an output file in csv format for submission to
    Kaggle or AIcrowd.

    Args:
        ids (np.ndarray): event ids associated with each prediction.
        y_pred (np.ndarray): predicted class labels.
        filename (str): name of csv output file to be created.
    """
    # Create output directory if needed
    create_out_dir()

    # Create csv submission
    with open(filename, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

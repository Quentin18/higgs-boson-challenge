# -*- coding: utf-8 -*-
"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False, label_b=-1):
    """
    Loads data and returns y (class labels), tX (features)
    and ids (event ids).

    The label_b argument must be -1 or 0 (default: -1).
    """
    y = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=str,
                      usecols=1)
    x = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = label_b

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data, label_b=-1):
    """Generates class predictions given weights, and a test data matrix.

    The label_b argument must be -1 or 0 (default: -1).
    """
    border=(label_b+1)/2
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= border)] = -1
    y_pred[np.where(y_pred > border)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


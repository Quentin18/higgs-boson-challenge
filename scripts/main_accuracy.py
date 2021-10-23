"""
Compute the accuracy score of a classification algorithm,
and show confusion matrix.
"""
# flake8: noqa: E402
import numpy as np

from path import add_src_to_path, extract_archives, DATA_TRAIN_PATH
from proj1_helpers import load_csv_data, predict_labels

# Add src to path to import implementations
add_src_to_path()

# Import functions from src
from data import standardize
from helpers import drop_columns_nan, split_data
from implementations import least_squares, logistic_regression
from metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

# Classifier to use
CLASSIFIER = 'logistic_regression'
# CLASSIFIER = 'least_squares'


def main():
    """
    Main function to compute the accuracy of classifiers.
    """
    # Extract archives if needed
    extract_archives()

    # Load the data
    print('[1/7] Load data')
    y, x, _ = load_csv_data(DATA_TRAIN_PATH, sub_sample=False, label_b=0)

    # Drop columns with -999 and standardize data
    print('[2/7] Clean data')
    x = drop_columns_nan(x)
    x = standardize(x)

    # Split data
    split_ratio = 0.8
    print(f'[3/7] Split data (ratio: {split_ratio})')
    x_tr, x_te, y_tr, y_te = split_data(x, y, split_ratio)

    print('[4/7] Run classification algorithm')

    if CLASSIFIER == 'least_squares':
        w, _ = least_squares(y_tr, x_tr)

    elif CLASSIFIER == 'logistic_regression':
        x_te = np.c_[np.ones((y_te.shape[0], 1)), x_te]
        initial_w = np.zeros((x_tr.shape[1], 1))
        max_iters = 10000
        gamma = 1e-6
        threshold = 1e-8
        w, _ = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma,
                                threshold, info=True, agd=True)

    else:
        print('Unknown classifier')
        return

    # Predict
    print('[5/7] Predict')
    y_pred = predict_labels(w, x_te, label_b_in=0, label_b_out=0)

    # Confusion matrix
    print('[6/7] Confusion matrix:')
    conf_matrix = confusion_matrix(y_te, y_pred)
    print(conf_matrix)

    # Accuracy
    print(f'[7/7] Accuracy score: {accuracy_score(y_te, y_pred): .2f}')

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)


if __name__ == '__main__':
    main()

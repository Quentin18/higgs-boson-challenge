"""
Generate predictions for AIcrowd.
"""
# flake8: noqa: E402
import os
import numpy as np

from path import (add_src_to_path, create_out_dir, extract_archives,
                  DATA_TEST_PATH, DATA_TRAIN_PATH, OUT_DIR)
from proj1_helpers import create_csv_submission, load_csv_data, predict_labels

# Add src to path to import implementations
add_src_to_path()

# Import functions from src
from data import standardize
from implementations import least_squares, logistic_regression

# Classifier to use
CLASSIFIER = 'logistic_regression'
# CLASSIFIER = 'least_squares'

# Output file path
OUTPUT_PATH = os.path.join(OUT_DIR, 'predictions_test.csv')


def main():
    """
    Main function to generate predictions for AIcrowd.
    """
    # Extract archives if needed
    extract_archives()

    # Create output directory if needed
    create_out_dir()

    # Load the train data
    print('[1/7] Load train data')
    y, x, ids = load_csv_data(DATA_TRAIN_PATH, label_b=0)

    # Standardize train data
    print('[2/7] Clean train data')
    # x = drop_columns_nan(x)
    x = standardize(x)

    # Load the test data
    print('[3/7] Load test data')
    y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH, label_b=0)

    # Standardize test data
    print('[4/7] Clean test data')
    x_test = standardize(x_test)

    print('[5/7] Run classification algorithm')

    if CLASSIFIER == 'least_squares':
        w, _ = least_squares(y, x)

    elif CLASSIFIER == 'logistic_regression':
        x_test = np.c_[np.ones((y_test.shape[0], 1)), x_test]
        initial_w = np.zeros((x.shape[1], 1))
        max_iters = 10000
        gamma = 1e-6
        threshold = 1e-8
        w, _ = logistic_regression(y, x, initial_w, max_iters, gamma,
                                   threshold, info=True, agd=True)

    else:
        print('Unknown classifier')
        return

    # Generate predictions
    print('[6/7] Generate predictions')
    y_pred = predict_labels(w, x_test, label_b_in=0, label_b_out=-1)

    # Create submission
    print('[7/7] Create submission')
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

    print(f'File {OUTPUT_PATH} created')


if __name__ == '__main__':
    main()

"""
Generate predictions for AIcrowd.

Usage:

python3 run.py

The csv file produced will be "out/predictions.csv".
"""
# flake8: noqa: E402
import os

import numpy as np

# Import functions from scripts/
from csv_utils import create_csv_submission, load_csv_data
from path import (add_src_to_path, load_json_parameters, DATA_TEST_PATH,
                  DATA_TRAIN_PATH, OUT_DIR)

# Add src/ to path to import functions
add_src_to_path()

# Import functions from src/
from clean_data import clean_train_test_data_by_jet
from cross_validation import build_poly
from implementations import ridge_regression
from split_data import split_by_jet
from helpers import predict_labels
from print_utils import (print_shapes, print_shapes_by_jet, print_subset_label,
                         NB_SUBSETS)

# Classifier to use
CLASSIFIER = 'ridge_regression'

# Output file path
OUTPUT_PATH = os.path.join(OUT_DIR, 'predictions.csv')


def main():
    """
    Main function to generate predictions for AIcrowd.
    """
    # Load the train data
    print('[1/8] Load train data')
    y_tr, x_tr, ids_tr = load_csv_data(DATA_TRAIN_PATH)
    print_shapes(y_tr, x_tr)

    # Load the test data
    print('[2/8] Load test data')
    y_te, x_te, ids_te = load_csv_data(DATA_TEST_PATH)
    print_shapes(y_te, x_te)

    # Split and clean train data
    print('[3/8] Split train and test data by jet')
    y_tr_by_jet, x_tr_by_jet, _ = split_by_jet(y_tr, x_tr, ids_tr)
    y_te_by_jet, x_te_by_jet, ids_te_by_jet = split_by_jet(y_te, x_te, ids_te)

    print('[4/8] Clean train and test data')
    y_tr_by_jet, x_tr_by_jet, y_te_by_jet, x_te_by_jet = \
        clean_train_test_data_by_jet(y_tr_by_jet, x_tr_by_jet,
                                     y_te_by_jet, x_te_by_jet)
    print('Train:')
    print_shapes_by_jet(y_tr_by_jet, x_tr_by_jet)
    print('Test:')
    print_shapes_by_jet(y_te_by_jet, x_te_by_jet)

    # Load parameters
    print('[5/7] Load parameters')
    params = load_json_parameters()
    print(params[CLASSIFIER])
    lambda_ = params[CLASSIFIER]['lambda_']
    degrees = params[CLASSIFIER]['degree']

    # Run ridge regression
    print('[6/8] Run classification algorithm')
    w_by_jet = list()

    for i in range(NB_SUBSETS):
        print_subset_label(i)

        # Get train subset
        x_tr_jet, y_tr_jet = x_tr_by_jet[i], y_tr_by_jet[i]

        # Build polynomial basis
        phi_tr_jet = build_poly(x_tr_jet, degrees[i])

        # Run ridge regression
        w, loss = ridge_regression(y_tr_jet, phi_tr_jet, lambda_[i])
        print(f'Loss = {loss:.3f}')

        # Add weights to list
        w_by_jet.append(w)

    # Generate predictions
    print('[7/8] Generate predictions')

    y_pred_by_jet = list()

    for i in range(NB_SUBSETS):
        # Get subset
        x_te_jet, w = x_te_by_jet[i], w_by_jet[i]

        # Build polynomial basis
        phi_te_jet = build_poly(x_te_jet, degrees[i])

        # Predict labels
        y_pred = predict_labels(w, phi_te_jet)
        y_pred_by_jet.append(y_pred)

    # Create submission
    print('[8/8] Create submission')

    y_pred = np.concatenate(y_pred_by_jet)
    ids_pred = np.concatenate(ids_te_by_jet)

    create_csv_submission(ids_pred, y_pred, OUTPUT_PATH)

    print(f'File {OUTPUT_PATH} created')


if __name__ == '__main__':
    main()

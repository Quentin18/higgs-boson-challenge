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
from implementations import least_squares, logistic_regression
from split_data import split_by_jet
from helpers import merge_y
from metrics import get_proportions

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

    # Clean train data
    print('[2/7] Clean train data')

    JET_COL = 22
    MAX_JET = 3
    y_by_jet, x_by_jet = split_by_jet(y, x, JET_COL, MAX_JET, clean=True)


    # Load the test data
    print('[3/7] Load test data')
    y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH, label_b=0)
    

    # Clean test data
    print('[4/7] Clean test data')
    y_by_jet_test, x_by_jet_test, ind_by_jet = split_by_jet(y_test, x_test, JET_COL, MAX_JET, clean=True,test=True)

    print('[5/7] Run classification algorithm')

    if CLASSIFIER == 'least_squares':
        w_by_jet = list()
        for i, x_tr_jet, y_tr_jet in zip(
            range(len(x_by_jet)), x_by_jet, y_by_jet):
            print(f'Jet = {i}:')
            w, _ = least_squares(y_tr_jet,x_tr_jet)
            w_by_jet.append(w)

    elif CLASSIFIER == 'logistic_regression':
        max_iters = 5000
        gamma = 1e-5
        threshold = 1e-8
        w_by_jet = list()
        for i, x_tr_jet, y_tr_jet in zip(
            range(len(x_by_jet)), x_by_jet, y_by_jet):
            print(f'Jet = {i}:')
            initial_w = np.zeros((x_tr_jet.shape[1], 1))
            w, _ = logistic_regression(y_tr_jet, x_tr_jet, initial_w,
                                max_iters, gamma, threshold,
                                info=True, sgd=False)
            w_by_jet.append(w)
    else:
        print('Unknown classifier')
        return

    # Generate predictions
    print('[6/7] Generate predictions')
    y_pred_by_jet = list()
    for i, w, x_te_jet, y_te_jet in zip(
        range(len(x_by_jet)), w_by_jet,x_by_jet_test, y_by_jet_test):
        x_te_jet = np.c_[np.ones((y_te_jet.shape[0], 1)), x_te_jet]
        y_pred = predict_labels(w, x_te_jet, label_b_in=0, label_b_out=-1,
                            use_sigmoid=True)
        y_pred_by_jet.append(y_pred)
    

    # Create submission
    print('[7/7] Create submission')
    y_pred=merge_y(y_pred_by_jet,ind_by_jet)
    proportions = get_proportions(y_pred)
    print('Proportions:', proportions)
    
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

    print(f'File {OUTPUT_PATH} created')


if __name__ == '__main__': 
    main()
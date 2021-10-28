"""
Compute the accuracy score of a classification algorithm,
and show confusion matrix.
"""
# flake8: noqa: E402
import matplotlib.pyplot as plt
import numpy as np

# Import functions from src/
from path import (DATA_TRAIN_PATH, add_src_to_path, create_out_dir,
                  extract_archives)
from proj1_helpers import load_csv_data

# Add src/ to path to import implementations
add_src_to_path()

# Import functions from scripts/
from clean_data import clean_data_by_jet
from helpers import predict_labels
from implementations import (least_squares, logistic_regression,
                             reg_logistic_regression)
from metrics import accuracy_score, confusion_matrix
from plot_utils import plot_accuracies, plot_confusion_matrix
from print_utils import print_shapes, print_shapes_by_jet
from split_data import split_by_jet, split_train_test

# Classifier to use
# CLASSIFIER = 'least_squares'
# CLASSIFIER = 'logistic_regression'
CLASSIFIER = 'reg_logistic_regression'


def main():
    """
    Main function to compute the accuracy of classifiers.
    """
    # Extract archives if needed
    extract_archives()

    # Create output directory if needed
    create_out_dir()

    # Load the data
    print('[1/7] Load data')
    y, x, _ = load_csv_data(DATA_TRAIN_PATH, label_b=0)
    print_shapes(y, x)

    # Split data train test
    print('[2/7] Split data train/test')
    x_tr, x_te, y_tr, y_te = split_train_test(y, x)

    print('[3/7] Split and clean train data by jet')
    # Split train data by jet
    y_tr_by_jet, x_tr_by_jet, _ = split_by_jet(y_tr, x_tr)

    # Clean train data by jet
    cols_to_remove_by_jet = clean_data_by_jet(y_tr_by_jet, x_tr_by_jet, k=0)
    print_shapes_by_jet(y_tr_by_jet, x_tr_by_jet)

    print('[4/7] Split and clean test data by jet')
    # Split test data by jet
    y_te_by_jet, x_te_by_jet, _ = split_by_jet(y_te, x_te)

    # Clean test data by jet
    clean_data_by_jet(y_te_by_jet, x_te_by_jet, cols_to_remove_by_jet)
    print_shapes_by_jet(y_te_by_jet, x_te_by_jet)

    print('[5/7] Run classification algorithm')

    w_by_jet = list()
    if CLASSIFIER == 'least_squares':
        for i, x_tr_jet, y_tr_jet in zip(range(len(x_tr_by_jet)),
                                         x_tr_by_jet, y_tr_by_jet):
            print(f'Jet = {i}:')
            w, _ = least_squares(y_tr_jet, x_tr_jet)
            w_by_jet.append(w)

    elif CLASSIFIER == 'logistic_regression':
        max_iters = 5000
        gamma = 1e-5
        threshold = 1e-8
        for i, x_tr_jet, y_tr_jet in zip(range(len(x_tr_by_jet)),
                                         x_tr_by_jet, y_tr_by_jet):
            print(f'Jet = {i}:')
            initial_w = np.zeros((x_tr_jet.shape[1], 1))
            w, _ = logistic_regression(y_tr_jet, x_tr_jet, initial_w,
                                       max_iters, gamma, threshold,
                                       info=True, agd=True)
            w_by_jet.append(w)

    elif CLASSIFIER == 'reg_logistic_regression':
        lambda_ = 1e-3
        max_iters = 5000
        gamma = 1e-5
        threshold = 1e-8
        for i, x_tr_jet, y_tr_jet in zip(range(len(x_tr_by_jet)),
                                         x_tr_by_jet, y_tr_by_jet):
            print(f'Jet = {i}:')
            initial_w = np.zeros((x_tr_jet.shape[1], 1))
            w, _ = reg_logistic_regression(y_tr_jet, x_tr_jet, lambda_,
                                           initial_w, max_iters, gamma,
                                           threshold, info=True)
            w_by_jet.append(w)

    else:
        print('Unknown classifier')
        return

    # Predict test data by jet
    plt.figure(figsize=(8, 8))
    y_pred_by_jet = list()
    accuracies = list()

    use_sigmoid = CLASSIFIER in ['logistic_regression',
                                 'reg_logistic_regression']

    for i, w, x_te_jet, y_te_jet in zip(
        range(len(y_te_by_jet)), w_by_jet,x_te_by_jet, y_te_by_jet):
        print(f'Jet = {i}:')
        # Predict labels
        if use_sigmoid:
            x_te_jet = np.c_[np.ones((y_te_jet.shape[0], 1)), x_te_jet]
        y_pred = predict_labels(w, x_te_jet, label_b_in=0, label_b_out=0,
                                use_sigmoid=use_sigmoid)
        y_pred_by_jet.append(y_pred)

        # Accuracy score
        accuracy = accuracy_score(y_te_jet, y_pred)
        accuracies.append(accuracy)
        print(f'Accuracy score: {accuracy:.2f}')

        # Confusion matrix
        ax = plt.subplot(2, 2, i + 1)
        conf_matrix = confusion_matrix(y_te, y_pred)
        plot_confusion_matrix(conf_matrix, ax=ax, title=f'Jet = {i}')

    plt.suptitle('Confusion matrices')
    plt.tight_layout()
    plt.show()

    # Plot accuracies
    plot_accuracies(accuracies)
    plt.show()

    # Concatenate results
    y_te = np.concatenate(y_te_by_jet)
    y_pred = np.concatenate(y_pred_by_jet)

    # Global accuracy
    print(f'Global accuracy score: {accuracy_score(y_te, y_pred):.2f}')

    # Global confusion matrix
    conf_matrix = confusion_matrix(y_te, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)


if __name__ == '__main__':
    main()

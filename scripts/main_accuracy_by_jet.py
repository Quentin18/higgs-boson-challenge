"""
Compute the accuracy score of a classification algorithm,
and show confusion matrix.
"""
# flake8: noqa: E402
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Import functions from src/
from path import (add_src_to_path, load_json_parameters, DATA_TRAIN_PATH,
                  FIGS_DIR)
from csv_utils import load_csv_data

# Add src/ to path to import implementations
add_src_to_path()

# Import functions from scripts/
import implementations
from clean_data import clean_train_test_data_by_jet
from cross_validation import build_poly
from helpers import predict_labels
from metrics import accuracy_score, confusion_matrix
from plot_utils import plot_accuracies, plot_confusion_matrix
from print_utils import (get_subset_label, print_shapes, print_shapes_by_jet,
                         print_subset_label)
from split_data import split_by_jet, split_train_test, NB_SUBSETS

CLASSIFIERS = [
    'gradient_descent',
    'stochastic_gradient_descent',
    'least_squares',
    'ridge_regression',
    'logistic_regression',
    'regularized_logistic_regression'
]


def main():
    """
    Main function to compute the accuracy of classifiers.
    """
    # Parse args from command line
    parser = argparse.ArgumentParser(
        description='Compute the accuracy of a classifier.')
    parser.add_argument(
        '--clf', choices=CLASSIFIERS,
        help='classifier to use (default: ridge_regression)',
        default='ridge_regression')
    args = parser.parse_args()

    clf = args.clf
    clf_name = clf.replace('_', ' ')

    is_gradient = 'gradient' in clf
    is_logistic = 'logistic' in clf
    need_polynomial_expansion = clf in ('least_squares', 'ridge_regression')
    label_b = 0 if is_logistic else -1

    # Load the data
    print('[1/7] Load data')
    y, x, _ = load_csv_data(DATA_TRAIN_PATH, label_b=label_b)
    print_shapes(y, x)

    # Split data train test
    print('[2/7] Split data train and test')
    x_tr, x_te, y_tr, y_te = split_train_test(y, x)

    # Split and clean train data
    print('[3/7] Split train and test data by jet')
    y_tr_by_jet, x_tr_by_jet, _ = split_by_jet(y_tr, x_tr)
    y_te_by_jet, x_te_by_jet, _ = split_by_jet(y_te, x_te)

    print('[4/7] Clean train and test data')
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
    clf_params = params.get(clf)
    if clf_params is None:
        print('No parameters found. Exit.')
        return
    print(clf_params)

    print(f'[6/7] Run {clf_name}')

    if need_polynomial_expansion:
        degrees = clf_params['degree']
        del clf_params['degree']

    if 'lambda_' in clf_params and isinstance(clf_params['lambda_'], list):
        lambdas = clf_params['lambda_']
        del clf_params['lambda_']
    else:
        lambdas = None

    w_by_jet = list()
    for i in range(NB_SUBSETS):
        print_subset_label(i)

        # Get train subset
        x_tr_jet, y_tr_jet = x_tr_by_jet[i], y_tr_by_jet[i]

        if is_logistic or is_gradient:
            clf_params['initial_w'] = np.zeros((x_tr_jet.shape[1], 1))

        if lambdas:
            clf_params['lambda_'] = lambdas[i]

        if need_polynomial_expansion:
            # Build polynomial basis
            x_tr_jet = build_poly(x_tr_jet, degrees[i])

        # Run algorithm on train data
        if clf == 'gradient_descent':
            w, _ = implementations.least_squares_GD(
                y_tr_jet, x_tr_jet, **clf_params)

        elif clf == 'stochastic_gradient_descent':
            w, _ = implementations.least_squares_SGD(
                y_tr_jet, x_tr_jet, **clf_params)

        elif clf == 'least_squares':
            w, _ = implementations.least_squares(
                y_tr_jet, x_tr_jet, **clf_params)

        elif clf == 'ridge_regression':
            w, _ = implementations.ridge_regression(
                y_tr_jet, x_tr_jet, **clf_params)

        elif clf == 'logistic_regression':
            w, _ = implementations.logistic_regression(
                y_tr_jet, x_tr_jet, **clf_params)

        elif clf == 'regularized_logistic_regression':
            w, _ = implementations.reg_logistic_regression(
                y_tr_jet, x_tr_jet, **clf_params)

        else:
            print('Unknown classifier. Exit.')
            return

        # Add weights to list
        w_by_jet.append(w)

    # Generate predictions
    print('[7/8] Generate predictions')

    y_pred_by_jet = list()
    accuracies = list()
    conf_matrices = list()

    for i in range(NB_SUBSETS):
        print_subset_label(i)

        # Get subset
        x_te_jet, y_te_jet, w = x_te_by_jet[i], y_te_by_jet[i], w_by_jet[i]

        # Build polynomial basis
        if need_polynomial_expansion:
            x_te_jet = build_poly(x_te_jet, degrees[i])

        elif is_logistic:
            x_te_jet = np.c_[np.ones((y_te_jet.shape[0], 1)), x_te_jet]

        # Predict labels
        y_pred = predict_labels(w, x_te_jet, label_b=label_b,
                                use_sigmoid=is_logistic)
        y_pred_by_jet.append(y_pred)

        # Accuracy score
        accuracy = accuracy_score(y_te_jet, y_pred)
        accuracies.append(accuracy)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_te_jet, y_pred)
        conf_matrices.append(conf_matrix)

    # Concatenate results
    y_te = np.concatenate(y_te_by_jet)
    y_pred = np.concatenate(y_pred_by_jet)

    # Global accuracy
    global_accuracy = accuracy_score(y_te, y_pred)
    print(f'Accuracy = {global_accuracy:.4f}')

    # Global confusion matrix
    global_conf_matrix = confusion_matrix(y_te, y_pred)
    plot_confusion_matrix(global_conf_matrix)

    # Plot confusion matrices
    plt.figure(figsize=(8, 8))

    # Plot subset matrices
    for i in range(NB_SUBSETS):
        ax = plt.subplot(2, 2, i + 1)
        plot_confusion_matrix(conf_matrices[i], ax=ax,
                              title=get_subset_label(i))

    # Plot global matrix
    ax = plt.subplot(2, 2, 4)
    plot_confusion_matrix(global_conf_matrix, ax=ax, title='Global')

    plt.suptitle(f'Confusion matrices with {clf_name}')
    plt.tight_layout()

    # Save figure
    path = os.path.join(FIGS_DIR, f'confusion_matrix_{clf}.pdf')
    plt.savefig(path)

    # Plot accuracies
    plot_accuracies(accuracies + [global_accuracy],
                    title=f'Accuracies with {clf_name}')
    plt.tight_layout()

    # Save figure
    path = os.path.join(FIGS_DIR, f'accuracy_{clf}.pdf')
    plt.savefig(path)

    plt.show()


if __name__ == '__main__':
    main()

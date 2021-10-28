"""
Print utils.
"""
import numpy as np

from split_data import NB_SUBSETS

SUBSET_LABELS = ('JET = 0', 'JET = 1', 'JET >= 2')


def print_shapes(y: np.ndarray, x: np.ndarray) -> None:
    """Prints the shapes of y and x.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
    """
    print('Shape of y:', y.shape)
    print('Shape of x:', x.shape)


def print_shapes_by_jet(y_by_jet: list, x_by_jet: list) -> None:
    """Prints the shapes of y and x by jet.

    Args:
        y_by_jet (list): y vectors by jet.
        x_by_jet (list): x matrices by jet.
    """
    for i in range(NB_SUBSETS):
        print_subset_label(i)
        print_shapes(y_by_jet[i], x_by_jet[i])


def get_subset_label(i: int) -> str:
    """Returns the label of subset i.

    Args:
        i (int): index (0, 1 or 2).

    Returns:
        str: label of subset i.
    """
    return SUBSET_LABELS[i]


def print_subset_label(i: int) -> None:
    """Prints the label of subset i.

    Args:
        i (int): index (0, 1 or 2).
    """
    print(get_subset_label(i) + ':')

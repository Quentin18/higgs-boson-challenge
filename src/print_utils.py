"""
Print utils.
"""
import numpy as np


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
    for i, y, x in zip(range(len(x_by_jet)), y_by_jet, x_by_jet):
        print(f'Jet = {i}:')
        print_shapes(y, x)

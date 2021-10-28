"""
Plot utils using matplotlib.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from print_utils import SUBSET_LABELS
from split_data import split_by_label


def plot_counts(y: np.ndarray, title: str = 'Proportions', ax=None,
                show: bool = False) -> None:
    """Plots counts of values of a vector as a bar chart.

    Args:
        y (np.ndarray): vector.
        title (str, optional): title of the figure. Defaults to 'Proportions'.
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    uniques, counts = np.unique(y, return_counts=True)
    ax.bar(uniques, counts, tick_label=('s', 'b'),
           color=mcolors.TABLEAU_COLORS)
    ax.set_title(title)

    # Show plot
    if show:
        plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, cmap: str = 'bwr',
                          ax=None, show: bool = False,
                          title: str = 'Confusion matrix') -> None:
    """Plots a confusion matrix.

    Args:
        conf_matrix (np.ndarray): confusion matrix.
        cmap (str, optional): colormap recognized by matplotlib.
        Defaults to 'bwr'.
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
        title (bool, optional): title of the confusion matrix.
        Defaults to 'Confusion Matrix'.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Plot matrix
    ax.matshow(conf_matrix, cmap=cmap, alpha=0.3)

    # Add values as text
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center',
                    size='x-large')

    # Set labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)

    # Show plot
    if show:
        plt.show()


def plot_accuracies(accuracies: list, title: str = 'Accuracies') -> None:
    """Plots the accuracies.

    Args:
        accuracies (list): list of accuracies.
        title (bool, optional): title of the accuracies plot.
        Defaults to 'Accuracies'.
    """
    # Get tick labels
    length = len(accuracies)
    if length > 3:
        tick_label = SUBSET_LABELS + ('Global',)
    else:
        tick_label = SUBSET_LABELS

    # Plot bars
    plt.figure(figsize=(6, 4))
    x = list(range(length))
    plt.bar(x, accuracies, tick_label=tick_label, color=mcolors.TABLEAU_COLORS)

    # Add values as text
    for index, value in enumerate(accuracies):
        plt.text(index - 0.15, value + 1e-2, str(round(value, 3)))

    plt.ylim(top=1)
    plt.title(title)


def scatter(x1: np.ndarray, x2: np.ndarray, y: np.ndarray, label_b=0,
            title=None, ylabel=None, xlabel=None, ax=None,
            show: bool = False) -> None:
    """scatter plot of x1 and x2 with respect to label

        Args:
        x1 (np.ndarray)
        x2 (np.ndarray)
        y (np.ndarray)
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
    """
    x1_s, x1_b = split_by_label(y, x1, label_b, plot=True)
    x2_s, x2_b = split_by_label(y, x2, label_b, plot=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    plt.scatter(x1_b, x2_b, color='red', alpha=1, label="Not boson", s=1)
    plt.scatter(x1_s, x2_s, color='blue', alpha=0.3, label="boson", s=1)
    plt.legend()

    # Decorate
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if show:
        plt.show()


def scatter_all_features(x: np.ndarray, y: np.ndarray, label_b=0) -> None:
    """Plot all features in a scatter plot

        Args:
        x (np.ndarray)
        y (np.ndarray)
    """
    nb_features = x.shape[1]
    if nb_features % 2 == 1:
        nb_features += 1
        x = np.c_[x, x[:, -2]]
    j = 0
    for i in range(0, nb_features-1, 2):
        j += 1
        scatter(x[:, i], x[:, i+1], y, label_b=0, xlabel="Feature %d" % i,
                ylabel="Feature %d" % (i+1))

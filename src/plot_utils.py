"""
Plot utils using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_counts(y: np.ndarray, title: str = None, ax=None,
                show: bool = False) -> None:
    """Plots counts of values of a vector as a bar chart.

    Args:
        y (np.ndarray): vector.
        title (str, optional): title of the figure. Defaults to None.
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    uniques, counts = np.unique(y, return_counts=True)
    ax.bar(uniques, counts, tick_label=['s', 'b'])
    if title is not None:
        ax.set_title(title)

    # Show plot
    if show:
        plt.show()


def plot_confusion_matrix(conf_matrix: np.ndarray, cmap: str = 'viridis',
                          ax=None, show: bool = False) -> None:
    """Plots a confusion matrix.

    Args:
        conf_matrix (np.ndarray): confusion matrix.
        cmap (str, optional): colormap recognized by matplotlib.
        Defaults to 'viridis'.
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Plot matrix
    ax.matshow(conf_matrix, cmap=cmap, alpha=0.3)

    # Add values as text
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center',
                    size='xx-large')

    # Set labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    # Show plot
    if show:
        plt.show()

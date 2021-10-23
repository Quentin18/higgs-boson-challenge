"""
Score and performance functions.
"""
import matplotlib.pyplot as plt
import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy classification score.

    Args:
        y_true (np.ndarray): correct target values.
        y_pred (np.ndarray): estimated targets as returned by a classifier.

    Returns:
        float: proportion of correctly classified samples.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes a confusion matrix.

    Args:
        y_true (np.ndarray): correct target values.
        y_pred (np.ndarray): estimated targets as returned by a classifier.

    Returns:
        np.ndarray: Confusion matrix whose i-th row and j-th column entry
        indicates the number of samples with true label being i-th class and
        predicted label being j-th class.
    """
    k = len(np.unique(y_true))  # Number of classes
    result = np.zeros((k, k), dtype=int)
    for i, j in zip(y_true, y_pred):
        result[i, j] += 1
    return result


def plot_confusion_matrix(conf_matrix: np.ndarray,
                          cmap: str = 'viridis') -> None:
    """Plot Confusion Matrix.

    Args:
        conf_matrix (np.ndarray): confusion matrix.
        cmap (str, optional): colormap recognized by matplotlib.
        Defaults to 'viridis'.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

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
    fig.tight_layout()
    plt.show()

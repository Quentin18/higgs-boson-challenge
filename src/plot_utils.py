"""
Plot utils using matplotlib.
"""
import matplotlib.pyplot as plt
import numpy as np
from split_data import split_by_label


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

def scatter(x1: np.ndarray, x2: np.ndarray,y: np.ndarray,label_b=0
                ,title=None,ylabel=None,xlabel=None, ax=None, show : bool = False) -> None:
    """scatter plot of x1 and x2 with respect to label

        Args:
        x1 (np.ndarray)
        x2 (np.ndarray)
        y (np.ndarray)
        ax (optional): matplotlib ax. Defaults to None.
        show (bool, optional): True to call plt.show. Defaults to True.
    """
    x1_s, x1_b=split_by_label(y, x1, label_b,plot=True)
    x2_s, x2_b=split_by_label(y, x2, label_b,plot=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    plt.scatter(x1_b,x2_b,color= 'red',alpha=1)
    plt.scatter(x1_s,x2_s,color='blue',alpha=0.5)
    

    # Decorate
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if show:
        plt.show() 

def scatter_all_features(x: np.ndarray ,y: np.ndarray,label_b=0)->None:
    """Plot all features in a scatter plot

        Args:
        x (np.ndarray)
        y (np.ndarray)
    """
    nb_features=x.shape[1]
    if nb_features % 2 == 1:
        nb_features+=1
        x=np.c_[x,x[:,-2]]
    nb_plot=nb_features/2
    #fig, axs = plt.subplots(ncols=3,nrows=nb_features//3 + 1 ,figsize=(25, 60))
    j=0
    for i in range(0,nb_features-1,2):
        j+=1
        #plt.subplot(3,nb_features//3 + 1,j)
        scatter(x[:,i],x[:,i+1],y,label_b=0,xlabel="feature {i}",ylabel="feature {j}",title="feature {i} vs feature {j}")
    





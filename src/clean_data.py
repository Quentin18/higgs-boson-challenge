"""
Functions to clean data.
"""
import numpy as np


def standardize(x: np.ndarray) -> np.ndarray:
    """Outputs the matrix x after normalization.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: matrix x normalized.
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def drop_columns(x: np.ndarray, indices: list) -> np.ndarray:
    """Drops columns from x.

    Args:
        x (np.ndarray): matrix.
        indices (list): indices of columns to remove.

    Returns:
        np.ndarray: x with removed columns.
    """
    return np.delete(x, indices, axis=1)


def drop_columns_nan(x: np.ndarray, value: int = -999) -> np.ndarray:
    """Drops the columns from x containing a certain value.

    Args:
        x (np.ndarray): matrix.
        value (int, optional): value to remove. Defaults to -999.

    Returns:
        np.ndarray: matrix x witout columns containing value.
    """
    x = np.where(x == value, np.nan, x)     # replace value to nan
    x = x[:, ~np.isnan(x).any(axis=0)]      # drop columns with nan
    return x


def get_columns_all_same(x: np.ndarray) -> np.ndarray:
    """Returns the columns indices of x containing the same value on every row.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: columns indices.
    """
    indices = list()
    for col in range(x.shape[1]):
        unique = np.unique(x[:, col])
        if unique.size == 1:
            indices.append(col)
    return np.array(indices)


def get_columns_to_remove_by_jet(x_by_jet: list) -> list:
    """Returns the columns indices to remove by jet.

    A column needs to be removed if all its values are the same.

    Args:
        x_by_jet (list): x matrices by jet.

    Returns:
        list: columns indices to remove by jet.
    """
    return [get_columns_all_same(x) for x in x_by_jet]


def clean_data_by_jet(x_by_jet: list, std: bool = True) -> None:
    """Cleans the dataset by jet.

    It removes columns with same data and standardize the other columns.

    Args:
        x_by_jet (list): x matrices by jet.
        std (bool, optional): True to standardize data. Defaults to True.
    """
    cols_to_remove_by_jet = get_columns_to_remove_by_jet(x_by_jet)
    for i in range(len(x_by_jet)):
        # Remove columns if needed
        if cols_to_remove_by_jet[i].size > 0:
            x_by_jet[i] = drop_columns(x_by_jet[i], cols_to_remove_by_jet[i])
        # Standardize data
        if std:
            x_by_jet[i] = standardize(x_by_jet[i])


def replace_empty_with_mean(x: np.ndarray, value: int = -999) -> np.ndarray:
    """Replaces empty cells with mean of true values.

    Args:
        x (np.ndarray): matrix.
        value (int, optional): empty value. Defaults to -999.

    Returns:
        np.ndarray: x with empty values replaced by mean.
    """
    for i in range(x.shape[1]):
        ind_full_cell = np.where(x[:, i] != value)
        mean_feature = np.mean(x[ind_full_cell, i])
        ind_empty_cell = np.where(x[:, i] == value)
        x[ind_empty_cell, i] = mean_feature
    return x


def replace_nan_to_zero(x: np.ndarray) -> np.ndarray:
    """Replaces the nan values by zero in x.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: matrix x with zeros instead of nan.
    """
    x[np.isnan(x)] = 0
    return x


def under_sample(y: np.ndarray, x: np.ndarray) -> tuple:
    """Deletes instances from the over-represented class.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.

    Returns:
        tuple: y and x with deleted instances.
    """
    unique, counts = np.unique(y, return_counts=True)
    index_class_over_represented = np.argmax(counts)
    index_class_under_represented = np.argmin(counts)
    class_over_represented = unique[index_class_over_represented]
    indices = np.where(y == class_over_represented)[0]
    nb_to_remove = counts[index_class_over_represented] - \
        counts[index_class_under_represented]
    indices_to_remove = indices[:nb_to_remove]
    return (np.delete(y, indices_to_remove),
            np.delete(x, indices_to_remove, axis=0))

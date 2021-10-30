"""
Functions to clean data.
"""
import numpy as np

from stats_tests import anova_test


def get_mean_std(x: np.ndarray) -> tuple:
    """Returns the mean and std of a matrix by rows.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple: mean, std.
    """
    return np.mean(x, axis=0), np.std(x, axis=0)


def log_transform(x: np.ndarray, min_: float = None) -> np.ndarray:
    """Applies a log transform on the matrix.

    Args:
        x (np.ndarray): matrix.
        min_ (float, optional): minimum to scale values of x. Defaults to None.

    Returns:
        np.ndarray: matrix after log transform.
    """
    if min_ is None:
        min_ = np.min(x)
    if not np.all(x > 0):
        x = x - min_ + 1
    return np.log(x)


def standardize(x: np.ndarray, mean: np.ndarray = None,
                std: np.ndarray = None) -> np.ndarray:
    """Outputs the matrix x after normalization.

    Args:
        x (np.ndarray): matrix.
        mean (np.ndarray, optional): mean. Defaults to None.
        std (np.ndarray, optional): std. Defaults to None.

    Returns:
        np.ndarray: x after normalization.
    """
    if mean is None and std is None:
        mean, std = get_mean_std(x)
    return (x - mean) / std


def robust_scale(x: np.ndarray) -> np.ndarray:
    """Scales the matrix x using robust scale.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: x after robust scale.
    """
    median = np.median(x, axis=0)
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    return (x - median) / iqr


def replace_empty_with_median(x: np.ndarray, value: int = -999) -> np.ndarray:
    """Replaces empty cells with median of true values.

    Args:
        x (np.ndarray): matrix.
        value (int, optional): empty value. Defaults to -999.

    Returns:
        np.ndarray: x with empty values replaced by median.
    """
    for i in range(x.shape[1]):
        ind_full_cell = np.where(x[:, i] != value)
        median_feature = np.median(x[ind_full_cell, i])
        ind_empty_cell = np.where(x[:, i] == value)
        x[ind_empty_cell, i] = median_feature
    return x


def remove_outliers(y: np.ndarray, x: np.ndarray, k: int) -> tuple:
    """Removes the outliers from x and y.

    Args:
        y (np.ndarray): output desired values.
        x (np.ndarray): input data.
        k (int): threashold.

    Returns:
        tuple: x and y without outliers.
    """
    mu, sigma = np.mean(x, axis=0), np.std(x, axis=0, ddof=1)
    indices = np.all(np.abs((x - mu) / sigma) < k, axis=1)
    return y[indices], x[indices]


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
    return np.array(indices).astype(int)


def get_columns_useless_anova(y: np.ndarray, x: np.ndarray,
                              ind_col_anova: list = None,
                              k_anova: float = 1e-3,
                              label_b: int = -1) -> np.ndarray:
    """Returns the columns indices of x useless according to Anova.

    Args:
        y (np.ndarray): input data labels.
        x (np.ndarray): input data.
        k (float, optional): critique value for feature selection with
        anova test. Defaults to 1e-3.
        label_b (int, optional): label of "b" events. Defaults to 0.
        ind_col_anova (list): index of the columns of x

    Returns:
        np.ndarray: columns indices.
    """
    features, f = anova_test(y, x, label_b=label_b,
                             ind_col_anova=ind_col_anova)
    ind_to_remove = np.where(f < k_anova)
    features_to_remove = features[ind_to_remove]
    return features_to_remove.astype(int)


def clean_data(x: np.ndarray, cols_to_remove: np.ndarray = None,
               replace_empty: bool = True, log: bool = True, std: bool = True,
               rob_scale: bool = False) -> np.ndarray:
    """Cleans a dataset.

    - Removes columns.
    - Replaces empty cells with median.
    - Applies log transform.
    - Standardizes columns.

    Args:
        x (np.ndarray): input data.
        cols_to_remove (np.ndarray, optional): columns to remove.
        Defaults to None.
        replace_empty (bool, optional): True to replace empty data by median.
        Defaults to True.
        log (bool, optional): True to apply log transform. Defaults to True.
        std (bool, optional): True to standardize data. Defaults to True.
        rob_scale (bool, optional): True to robust scale data.
        Defaults to False.

    Returns:
        np.ndarray: cleaned data.
    """
    # Remove columns if needed
    if cols_to_remove is not None and cols_to_remove.size:
        x = drop_columns(x, cols_to_remove)

    # Replace empty cells with median
    if replace_empty:
        x = replace_empty_with_median(x)

    # Log transform
    if log:
        x = log_transform(x)

    # Standardize data
    if std:
        x = standardize(x)

    # Robust scale
    if rob_scale:
        x = robust_scale(x)

    return x


def get_columns_to_remove_by_jet(y_by_jet: list, x_by_jet: list,
                                 k_anova: float = None,
                                 label_b: int = -1) -> list:
    """Returns the columns indices to remove by jet.

    A column needs to be removed if:
    - All its values are the same.
    - The feature is useless according to Anova (optional).

    Args:
        y_by_jet (list): y labels by jet.
        x_by_jet (list): x matrices by jet.
        k_anova (float): critical value for anova filter. Defaults to None.
        label_b (int): label for anova test -1 or 0

    Returns:
        list: columns indices to remove by jet.
    """
    columns_to_remove = []
    for x, y in zip(x_by_jet, y_by_jet):
        cols = get_columns_all_same(x)
        if k_anova is not None:
            col_anova = list(set(range(0, 30))-set(cols))
            x_anova = np.delete(x, cols, axis=1)
            cols_anova = get_columns_useless_anova(y, x_anova,
                                                   ind_col_anova=col_anova,
                                                   k_anova=k_anova,
                                                   label_b=label_b)
            cols = np.concatenate((cols, cols_anova))
        columns_to_remove.append(cols)
    return columns_to_remove


def clean_data_by_jet(y_by_jet: list, x_by_jet: list,
                      cols_to_remove_by_jet: list = None,
                      replace_empty: bool = True, log: bool = True,
                      std: bool = True, rob_scale: bool = False,
                      k_anova: float = None, label_b: int = -1) -> list:
    """Cleans the dataset by jet.

    - Removes columns with same data.
    - Removes useless features according to Anova test (optional).
    - Replaces -999 by median (optional).
    - Log transform data (optional).
    - Standardizes columns (optional).

    Args:
        y_by_jet (list): y labels by jet.
        x_by_jet (list): x matrices by jet.
        cols_to_remove_by_jet (list, optional): list of columns to remove by
        jet. Defaults to None.
        replace_empty (bool, optional): True to replace empty data by median.
        Defaults to True.
        log (bool, optional): True to appy log transform. Defaults to True.
        std (bool, optional): True to standardize data. Defaults to True.
        rob_scale (bool, optional): True to robust scale data.
        Defaults to False.
        k_anova: (float, optional): critical values for Anova test.
        label_b: (int, optional): for anova test -1 or 1
        Defaults to None.

    Returns:
        list: columns to remove by jet.
    """
    # Get columns to remove
    if cols_to_remove_by_jet is None:
        cols_to_remove_by_jet = get_columns_to_remove_by_jet(
            y_by_jet, x_by_jet, k_anova=k_anova, label_b=label_b)

    # Clean data
    for i in range(len(x_by_jet)):
        x_by_jet[i] = clean_data(
            x_by_jet[i], cols_to_remove_by_jet[i], replace_empty, log=log,
            std=std, rob_scale=rob_scale)

    return cols_to_remove_by_jet


def clean_train_test_data_by_jet(y_tr_by_jet: list, x_tr_by_jet: list,
                                 y_te_by_jet: list, x_te_by_jet: list,
                                 replace_empty: bool = True,
                                 k_anova: float = None) -> tuple:
    """Cleans train and test data by jet.

    - Log transform with min of x train and x test.
    - Standardize with mean and std of train.

    Args:
        y_tr_by_jet (list): y train labels by jet.
        x_tr_by_jet (list): x train matrices by jet.
        y_te_by_jet (list): y test labels by jet.
        x_te_by_jet (list): x test matrices by jet.
        replace_empty (bool, optional): True to replace empty data by median.
        Defaults to True.
        k_anova: (float, optional): critical values for Anova test.
        Defaults to None.

    Returns:
        tuple: (y_tr_by_jet, x_tr_by_jet, y_te_by_jet, x_te_by_jet)
    """
    # Clean train data
    cols_to_remove_by_jet = clean_data_by_jet(
        y_tr_by_jet, x_tr_by_jet, replace_empty=replace_empty, log=False,
        std=False, k_anova=k_anova)

    # Clean test data
    clean_data_by_jet(y_te_by_jet, x_te_by_jet, cols_to_remove_by_jet,
                      replace_empty=replace_empty, log=False, std=False)

    # Log transform train and test
    for i in range(len(x_tr_by_jet)):
        if not (np.all(x_tr_by_jet[i] > 0) and np.all(x_te_by_jet[i] > 0)):
            min_ = min(np.min(x_tr_by_jet[i]), np.min(x_te_by_jet[i]))
        else:
            min_ = None
        x_tr_by_jet[i] = log_transform(x_tr_by_jet[i], min_)
        x_te_by_jet[i] = log_transform(x_te_by_jet[i], min_)

    # Standardize train and test using mean and std of train
    for i in range(len(x_tr_by_jet)):
        mean, std = get_mean_std(x_tr_by_jet[i])
        x_tr_by_jet[i] = standardize(x_tr_by_jet[i])
        x_te_by_jet[i] = standardize(x_te_by_jet[i], mean, std)

    return y_tr_by_jet, x_tr_by_jet, y_te_by_jet, x_te_by_jet

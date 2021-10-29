"""
Statistical tests.
"""
import numpy as np

from split_data import split_by_label


def anova_test(y: np.ndarray, x: np.ndarray, ind_col_anova: list = None,
               label_b: int = -1) -> tuple:
    """Anova test on features for features selection.

    Args:
        y (np.ndarray): input data features.
        x (np.ndarray): input data.
        label_b (int, optional): label of "b" event. Defaults to -1.
        ind_col_anova (list) : index of x columns

    Returns:
        tuple : (ind_F, F) with F f_value of features sorted,
        ind_F index of features
    """
    tX_a, tX_b = split_by_label(y, x, label_b)
    mean_labels = [np.mean(tX_a, 0), np.mean(tX_b, 0)]
    var_between_groups = np.var(mean_labels, 0)
    var_in_groups = np.var(tX_a, 0) + np.var(tX_b, 0)
    f = var_between_groups / var_in_groups
    F = np.argsort(f)
    for i in range(0, len(F)):
        F[i] = ind_col_anova[F[i]]
    return F, np.sort(f)

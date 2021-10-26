"""
Statistical tests.
"""
import numpy as np


def anova_test(tX_a, tX_b):
    """Anova test on features to see which one are relevant."""
    mean_labels = [np.mean(tX_a, 0), np.mean(tX_b, 0)]
    # Between the sum of square (variance between the mean of each groups)
    ssb = np.var(mean_labels, 0)
    # Within the sum of square (variance between)
    ssw = np.var(tX_a, 0) + np.var(tX_b, 0)
    f = ssb / ssw
    # print(np.argsort(f))
    return f

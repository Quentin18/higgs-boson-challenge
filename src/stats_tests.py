"""
Statistical tests.
"""
import numpy as np
from split_data import split_by_label


def anova_test(x : np.ndarray, y : np.ndarray, label_b=-1)-> list:
    """
        Anova test on features to see which one are relevant.
        
        Args:
        x (np.ndarray) : input data (data * features)
        y (np.ndarray) : input labels of x
        label_b (int) : label of b (0 or -1(default))

        Output:
        list of index of irrelevant features.   
    
    """
    tX_a,tX_b=split_by_label(y, x, label_b)
    mean_labels = [np.mean(tX_a, 0), np.mean(tX_b, 0)]
    # Between the sum of square (variance between the mean of each groups)
    ssb = np.var(mean_labels, 0)
    # Within the sum of square (variance between)
    ssw = np.var(tX_a, 0) + np.var(tX_b, 0)
    f = ssb / ssw
    print(np.argsort(f))
    return f

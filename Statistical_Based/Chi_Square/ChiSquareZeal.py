import numpy as np
from sklearn.feature_selection import chi2


def chi_square(X, y):
    """
    Chi square utilize the test of independence to assess whether the feature is independent of the class label.
    """
    F, pval = chi2(X, y)
    return F


def feature_ranking(F):

    idx = np.argsort(F)
    return idx[::-1]

from sklearn.feature_selection import VarianceThreshold


def Low_Variance_FS(X, threshold):
    sel = VarianceThreshold(threshold)
    # Features with a training-set variance lower than this threshold will be removed.
    # The default is to keep all features with non-zero variance,
    # i.e. remove the features that have the same value in all samples.
    return sel.fit_transform(X)

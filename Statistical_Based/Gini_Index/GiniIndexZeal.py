import numpy as np


def gini_index(X, y):
    n_samples, n_features = X.shape

    # initialize gini index for all features to be 0.5
    gini = np.ones(n_features) * 0.5

    # For i-th feature we define fi = x[:, i], v include all unique values in fi
    for i in range(n_features):  # for each feature
        v = np.unique(X[:, i])   # the number of different categories
        for j in range(len(v)):  #
            left_y = y[X[:, i] <= v[j]]
            right_y = y[X[:, i] > v[j]]

            # gini_left is sum of square of probability of occurrence of v[i] in left_y
            # gini_right is sum of square of probability of occurrence of v[i] in right_y
            gini_left = 0
            gini_right = 0

            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    # t1_left is probability of occurrence of k in left_y
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left

                if len(right_y) != 0:
                    # t1_right is probability of occurrence of k in left_y
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right

            gini_left = 1 - gini_left
            gini_right = 1 - gini_right

            # weighted average of len(left_y) and len(right_y)
            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

            # compute the gini index for the i-th feature
            value = np.true_divide(t1_gini, len(y))

            if value < gini[i]:
                gini[i] = value
    return gini


def feature_ranking(W):
    """
    Rank features in descending order according to their gini index values, the smaller the gini index,
    the most important the feature is
    """
    idx = np.argsort(W)
    return idx

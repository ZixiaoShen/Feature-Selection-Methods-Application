import numpy as np


def reliefF(X, y, **kwargs):
    """
    This function implements the reliefF feature selection
    :param X: {numpy array}, shape (n_samples, n_features) input data
    :param y: {numpy array}, shape (n_samples,) input class labels
    :param kwargs: {dictionary} parameters of reliefF:
           k {int} choices for the number of neighbors (default k = 5)
    :return: score: {numpy array}, shape reliefF score for each feature
    """
    


import scipy.io
import numpy as np

mat = scipy.io.loadmat('/home/zealshen/DATA/DATAfromASU/BiologicalData/colon.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape

X = X[:, 0:10]

# from skfeature.function.statistical_based import CFS
# idx = CFS.cfs(X_train, y)
# merit = CFS.merit_calculation(X_train, y)
#
# print(idx)
# print(merit)

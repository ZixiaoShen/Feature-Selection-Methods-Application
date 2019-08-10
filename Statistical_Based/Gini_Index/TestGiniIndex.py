import scipy.io
from sklearn import svm

mat = scipy.io.loadmat('/home/zealshen/DATA/DATAfromASU/BiologicalData/colon.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape

# from skfeature.function.statistical_based import gini_index
# score = gini_index.gini_index(X, y)
# idx = gini_index.feature_ranking(score)
# print(score)
# print(idx)

from Statistical_Based.Gini_Index import GiniIndexZeal
score = GiniIndexZeal.gini_index(X, y)
idx = GiniIndexZeal.feature_ranking(score)
print(score)
print(idx)


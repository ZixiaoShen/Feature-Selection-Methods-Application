import scipy.io
from sklearn import svm
from sklearn.model_selection import KFold
from skfeature.function.statistical_based import CFS

# import X and y
mat = scipy.io.loadmat('../Datasets/colon.mat')
X = mat['X'][:, 1:101]
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features

idx = CFS.cfs(X, y)
print(idx)
# split data into 10 folds
kf = KFold(n_splits=10)
for train_Id, test_Id in kf.split(X):
    print("Train:", train_Id, "Test:", test_Id)
    X_train, X_test = X[train_Id], X[test_Id]
    y_train, y_test = y[train_Id], y[test_Id]

    ## obtain the index of selected features on training set
    idx = CFS.cfs(X_train, y_train)
    print(idx)

import scipy.io
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from skfeature.function.statistical_based import CFS

# import X and y
mat = scipy.io.loadmat('../Datasets/colon.mat')
X = mat['X']
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features

clf = svm.LinearSVC()
correct = 0

# split data into 10 folds
kf = KFold(n_splits=10)
for train_Id, test_Id in kf.split(X):
    # print("Train:", train_Id, "Test:", test_Id)
    X_train, X_test = X[train_Id], X[test_Id]
    y_train, y_test = y[train_Id], y[test_Id]

    # obtain the index of selected features on training set
    idx = CFS.cfs(X_train, y_train)
    selected_features = X[:, idx]
    clf.fit(selected_features[train_Id], y[train_Id])
    y_predict = clf.predict(selected_features[test_Id])

    # obtain the classification accuracy on the test data
    acc = accuracy_score(y[test_Id], y_predict)
    correct = correct + acc

print('Accuracy:', float(correct)/10)

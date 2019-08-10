import scipy.io
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
#from skfeature.function.statistical_based import chi_square

mat = scipy.io.loadmat('../Datasets/BASEHOCK.mat')

X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape

from Statistical_Based.Chi_Square import ChiSquareZeal
score = ChiSquareZeal.chi_square(X, y)
idx = ChiSquareZeal.feature_ranking(score)

print(score)
print(idx)

# score = chi_square.chi_square(X, y)
# idx = chi_square.feature_ranking(score)
# print(idx)

# # perform evaluation on classification task
# num_fea = 100
# clf = svm.LinearSVC()
#
# correct = 0
# # split data into 10 folds
# kf = KFold(n_splits=10)
# for train_Id, test_Id in kf.split(X):
#     # print("Train:", train_Id, "Test:", test_Id)
#     X_train, X_test = X[train_Id], X[test_Id]
#     y_train, y_test = y[train_Id], y[test_Id]
#
#     # obtain the index of selected features on training set
#     score = chi_square.chi_square(X_train, y_train)
#
#     # rank feature in descending order according to score
#     idx = chi_square.feature_ranking(score)
#
#     # obtain the dataset on the selected features
#     selected_features = X[:, idx[0:num_fea]]
#
#     # train a classification model with the selected features on the training dataset
#     clf.fit(selected_features[train_Id], y[train_Id])
#
#     # predict the class labels of test data
#     y_predict = clf.predict(selected_features[test_Id])
#
#     # obtain the classification accuracy on the test data
#     acc = accuracy_score(y[test_Id], y_predict)
#     correct = correct + acc
#
# print('Accuracy:', float(correct)/10)

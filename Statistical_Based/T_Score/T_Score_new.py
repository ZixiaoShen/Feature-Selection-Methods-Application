import scipy.io
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm

mat = scipy.io.loadmat('/Users/shenzixiao/Dropbox/DATA/ASU/BiologicalData/colon.mat')
X = mat['X']  # data
x = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape  # number of samples and number of features

from Statistical_Based.T_Score import TscoreZeal
score = TscoreZeal.T_Score_FS(X, y)
idx = TscoreZeal.feature_ranking(score)
print("Score:", score)
print("idx:", idx)

# perform evaluation on classification task
num_fea = 100
clf = svm.LinearSVC()

correct = 0
# split data into 10 folds
kf = KFold(n_splits=10)
for train_Id, test_Id in kf.split(X):
    # print("Train:", train_Id, "Test:", test_Id)
    X_train, X_test = X[train_Id], X[test_Id]
    y_train, y_test = y[train_Id], y[test_Id]

    # obtain the index of selected features on training set
    score = TscoreZeal.T_Score_FS(X_train, y_train)

    # rank feature in descending order according to score
    idx = TscoreZeal.feature_ranking(score)

    # obtain the dataset on the selected features
    selected_features = X[:, idx[0:num_fea]]

    # train a classification model with the selected features on the training dataset
    clf.fit(selected_features[train_Id], y[train_Id])

    # predict the class labels of test data
    y_predict = clf.predict(selected_features[test_Id])

    # obtain the classification accuracy on the test data
    acc = accuracy_score(y[test_Id], y_predict)
    correct = correct + acc

print('Accuracy:', float(correct)/10)

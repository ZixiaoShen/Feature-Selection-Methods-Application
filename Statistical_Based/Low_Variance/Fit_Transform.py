# coding: utf-8
import pandas as pd
import numpy as np

# Create Feature Table
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                'Mitoses', 'Class']

# load the data from internet
data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
    names=column_names)

# replace the null
data = data.replace(to_replace='?', value=np.nan)
# drop the null data
data = data.dropna(how='any')

print(data.shape)

X = data[column_names[1:10]]
X = X.values
y = data[column_names[10]]
y = y.values

# Splitting the training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

y_train = pd.Series(y_train)
y_train.value_counts()

y_test = pd.Series(y_test)
y_test.value_counts()

from sklearn.preprocessing import StandardScaler
# normalize the data with the mean of 0 and variance of 1.
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# initialize Logistic Regression and SGD Classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()

# train the model using Logistic Regression
lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)

# train the model using SGD Classification
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)

from sklearn.metrics import classification_report
# get the accuracy results using logistic regression model score
print("Accuracy of LR Classifier:", lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

print("Accuracy of SGD Classifier:", sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))

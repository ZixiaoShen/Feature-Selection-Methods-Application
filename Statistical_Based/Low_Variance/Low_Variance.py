import scipy.io
#from skfeature.function.statistical_based import low_variance
from skfeature.utility import unsupervised_evaluation

mat = scipy.io.loadmat('../Datasets/BASEHOCK.mat')
X = mat['X']   # data
X = X.astype(float)
y = mat['Y']   # label
y = y[:, 0]

p = 0.1  # specify the threshold p to be 0.1
num_cluster = 2  # specify the number of clusters to be 2

# perform feature selection and obtain the dataset on the selected features
from Statistical_Based.Low_Variance.LowVarianceZeal import Low_Variance_FS
selected_features = Low_Variance_FS(X, p*(1-p))

# perform kmeans clustering based on the selected features and repeats 20 times
nmi_total = 0
acc_total = 0
for i in range(0, 20):
    nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=num_cluster, y=y)
    nmi_total += nmi
    acc_total += acc

# output the average NMI and average ACC
print('NMI:', float(nmi_total)/20)
print('ACC:', float(acc_total)/20)

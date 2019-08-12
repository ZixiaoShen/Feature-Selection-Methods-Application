import numpy as np
import scipy.io
from sklearn.metrics.pairwise import pairwise_distances

mat = scipy.io.loadmat('/home/zealshen/DATA/DATAfromASU/FaceImageData/COIL20.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape

distance = pairwise_distances(X, metric='manhattan')
score = np.zeros(n_features)

# the number of sampled instances is equal to the number of total instances
# for idx in range(n_samples):
idx = 1
near_hit = []
near_miss = dict()

self_fea = X[idx, :]
c = np.unique(y).tolist()

stop_dict = dict()
for label in c:
    stop_dict[label] = 0
del c[c.index(y[idx])]

p_dict = dict()
p_label_idx = float(len(y[y == y[idx]]))/float(n_samples)



for ele in near_hit:
    near_hit_term = np.array(abs(self_fea-X[ele, :])) + np.array(near_hit_term)


for (label, miss_list) in near_miss.items():
    near_miss_term[label] = np.zeros(n_features)
    for ele in miss_list:
        near_miss_term[label] = np.array(abs(self_fea-X[ele, :]))+np.array(near_miss_term[label])
    score += near_miss_term[label]/(k*p_dict[label])
score -= near_hit_term/k

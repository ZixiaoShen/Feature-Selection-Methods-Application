import scipy.io
from skfeature.function.similarity_based import reliefF

mat = scipy.io.loadmat('/Users/shenzixiao/Dropbox/DATA/ASU/FaceImageData/COIL20.mat')
X = mat['X']
X = X.astype(float)
y = mat['Y']
y = y[:, 0]
n_samples, n_features = X.shape

score = reliefF.reliefF(X, y)
idx = reliefF.feature_ranking(score)
print(idx)
print(score)

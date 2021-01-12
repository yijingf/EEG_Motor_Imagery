import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import *
from config import SUBs, freq_bands
from bandpower import *
from baseline import *
from sklearn.model_selection import train_test_split

# Preprocess config
l_freq = 4
h_freq = 30
resample_freq = 100 # original sfreq is 160

# load data
data_loader = DataLoader(window_len=100, overlap=0) # machine learning window_len 100, deep learning window_len: 20-40
X, y, _ = data_loader.load_data(SUBs, 
                                l_freq=l_freq, h_freq=h_freq, 
                                resample_sfreq=resample_freq, mesh=False)
X = np.array([getBandPower_Pool(x, freq_bands, resample_freq) for x in X])
# X is a matrix with the shape of (n, 3, 64) where n is the number of samples, 3 is the number of frequency bands and 64 is the number of channels

X = X.reshape((-1, 3*64))
# train_ratio = 0.60
# validation_ratio = 0.20
test_ratio = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_ratio/(1-test_ratio))

# default setting 
acc_dict = CV_Baselines(X_train, y_train, X_test, y_test, K = 5)

# grid searched parameters
pred_y_rf, acc_rf = RandForest_gridSearch(data = X_train, labels = y_train, test_data = X_test, test_labels = y_test, K = 3)
pred_y_svm, acc_svm = SVM_gridSearch(data = X_train, labels = y_train, test_data = X_test, test_labels = y_test, K = 3)

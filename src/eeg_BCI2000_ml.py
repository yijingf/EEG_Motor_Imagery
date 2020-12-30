import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import *
from config import SUBs, freq_bands

# Preprocess config
l_freq = 4
h_freq = 30
resample_freq = 100 # original sfreq is 160


# load data
data_loader = DataLoader(window_len=100, overlap=0)
X, Y, _ = data_loader.load_data(SUBs, l_freq=l_freq, h_freq=h_freq, resample_sfreq=resample_freq, mesh=False)
X = np.array([getBandPower_Pool(x, freq_bands, resample_freq) for x in X])

# X is a matrix with the shape of (n, 3, 64) where n is the number of samples, 3 is the number of frequency bands and 64 is the number of channels

# Todo: Split train/test/validation set


# train svm model
clf = svm.SVC(gamma='scale')
clf.fit(train_X, train_y)  

# predict validation set 
valid_pre_y = clf.predict(valid_X) 
val_acc = accuracy_score(valid_pre_y, valid_y)
print('valid accuracy: ', val_acc)
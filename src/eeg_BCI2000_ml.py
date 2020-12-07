import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import *
from config import SUBs

# Preprocess config
l_freq = 4
h_freq = 30
resample_freq = 100 # original sfreq is 160

# load data
data_loader = DataLoader(window_len=80, overlap=0)
(train_X, train_y), (valid_X, valid_y), test_set = data_loader.load_train_val_test(SUBs, l_freq=l_freq, h_freq=h_freq, resample_freq=resample_freq,
																				   mesh=False, one_hot=False)

# Todo: Feature Extraction
train_X = train_X.mean(axis=1)
valid_X = valid_X.mean(axis=1)

# train svm model
clf = svm.SVC(gamma='scale')
clf.fit(train_X, train_y)  

# predict validation set 
valid_pre_y = clf.predict(valid_X) 
val_acc = accuracy_score(valid_pre_y, valid_y)
print('valid accuracy: ', val_acc)
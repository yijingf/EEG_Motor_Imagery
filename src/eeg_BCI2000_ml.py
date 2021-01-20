import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import *
from config import SUBs, freq_bands
from bandpower import *
from baseline import *
from classical_ml import *
from sklearn.model_selection import train_test_split

# Preprocess config
l_freq = 4
h_freq = 30
resample_freq = 100 # original sfreq is 160

# load data
data_loader = DataLoader(window_len = 100, overlap = 0.5)
# machine learning window_len 100, deep learning window_len: 20-40
# for classical machine learning, we will use band power of three frequency bands (4~30HZ), so window_len should be long enough here
# take a window sufficiently long to encompasses at least two full cycles of the lowest frequency of interest. 
# https://raphaelvallat.com/bandpower.html 
# 4hz -> 0.25s -> 2 cycles 0.5s -> resample_freq=100: at least 50 samples

X, y, _ = data_loader.load_data(SUBs,
                                l_freq=l_freq, h_freq=h_freq,
                                resample_sfreq=resample_freq, mesh=False)
X = np.array([getBandPower_Pool(x, freq_bands, resample_freq) for x in X])
# X is a matrix with the shape of (n, 3, 64) where n is the number of samples
# 3 is the number of frequency bands and 64 is the number of channels

X = X.reshape((-1, 3*64))

# np.save('processed_X', X)
# np.save('processed_y', y)
# X = np.load('processed_X.npy')
# y = np.load('processed_y.npy')

acc_dict = gridSearch_baseline_models(X, y, test_ratio = 0.2, K = 3, verbose = 1)

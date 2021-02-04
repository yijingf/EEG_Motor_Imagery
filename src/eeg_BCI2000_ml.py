import sys
#from config import SUBs, freq_bands, split_ratio, resPath
from config import SUBs, freq_bands, split_ratio

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import * #will load dataDir and DataLoader()
from config import SUBs, freq_bands
from bandpower import * #will load getBandPower_Pool
from baseline import * #no grid search, classical machine learning models
from classical_ml import * #have grid search, will load gridSearch_baseline_models
from sklearn.model_selection import train_test_split
import os

# Preprocess config
l_freq = 4
h_freq = 30
resample_freq = 100 # original sfreq is 160

savedPath=r'D:\dataprocess\shared\EEG\data\MotorMovementImagery'
Xfilename=os.path.join(savedPath,'winLen100_overlap25_allseg_bandPower')
yfilename=os.path.join(savedPath,'winLen100_overlap25_allseg_label')

if not os.path.exists(Xfilename):
    # load data
    #data_loader = DataLoader(window_len = 100, overlap = 0.5) #50% overlapping
    data_loader = DataLoader(window_len = 100, overlap = 0.25)
    #data_loader = DataLoader(window_len = 100, overlap = 0) #no overlapping
    # machine learning window_len 100, deep learning window_len: 20-40
    # for classical machine learning, we will use band power of three frequency bands (4~30HZ), so window_len should be long enough here
    # # take a window sufficiently long to encompasses at least two full cycles of the lowest frequency of interest.
    # # https://raphaelvallat.com/bandpower.html
    # # 4hz -> 1 cycle: 1/4HZ=0.25s -> 2 cycles 0.5s -> resample_freq=100: 50 samples?

    X, y, _ = data_loader.load_data(SUBs,
                                    l_freq=l_freq, h_freq=h_freq,
                                    resample_sfreq=resample_freq, mesh=False)
    # np.save(Xfilename, X) #very slow and cost 2 times of your memory (99%memory), also the saved file costs 4G, so don't save here but save the bandpower
    # np.save(yfilename, y)
    print(X.shape) #(92290, 64, 100)

    X = np.array([getBandPower_Pool(x, freq_bands, resample_freq) for x in X]) #very slow!
    # X is a matrix with the shape of (n, 3, 64) where n is the number of samples
    # 3 is the number of frequency bands and 64 is the number of channels
    print(X.shape) #(92290, 3, 64)

    np.save(Xfilename, X) #save the bandpower
    np.save(yfilename, y) #save the label

else:
    X = np.load(Xfilename, mmap_mode='r')
    y = np.load(yfilename, mmap_mode='r')

X = X.reshape((-1, 3*64)) #convert to 2D, i.e., merge 3 frequency band powers across all 64 channels

# np.save('processed_X', X)
# np.save('processed_y', y)
# X = np.load('processed_X.npy')
# y = np.load('processed_y.npy')

# save results
org_stdout = sys.stdout
resfile = open(os.path.join(savedPath, 'ml_results.txt'), 'w')
sys.stdout = resfile

#acc_dict = gridSearch_baseline_models(X, y, test_ratio = 0.2, K = 3, verbose = 1) #
acc_dict = gridSearch_baseline_models(X, y, test_ratio = split_ratio[2], K = 3, verbose = 1)
sys.stdout = org_stdout
resfile.close()
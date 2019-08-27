import numpy as np
from tqdm import tqdm
from eeg_import import *
from sklearn.preprocessing import LabelBinarizer

def get_mesh(x):
    mesh = np.zeros((10, 11))
    mesh[0, 4:7] = x[21:24]
    mesh[1, 3:8] = x[24:29]
    mesh[2, 1:10] = x[29:38]
    mesh[3, 1:10] = x[[38] + list(range(7)) + [39]]
    mesh[4, :] = x[[42, 40] + list(range(8, 15)) + [41, 43]]
    mesh[5, 1:10] = x[[44] + list(range(14, 21)) + [45]]
    mesh[6, 1:10] = x[46:55]
    mesh[7, 3:8] = x[55:60]
    mesh[8, 4:7] = x[60:63]
    mesh[9, 5] = x[63]
    mesh = mesh.reshape((10, 11, 1))
    return mesh

def get_mesh_sequence(x, window_len=10):
    return np.array([get_mesh(x[:,j]) for j in range(window_len)])

def scale_data(X):
    for i in tqdm(range(64)):
        tmp = X[:,i,:]
        mean = np.mean(tmp)
        std = np.std(tmp)
        X[:,i,:] = (tmp - mean)/std
    return X

# def get_mesh_data(subj_num, window_len=10, labels=[0,1,2,3,4]):
#     X, y = get_data(subj_num)
#     X = transform(X, window_len)
#     lb = LabelBinarizer()
#     lb.fit(labels)
#     y = lb.transform(y)
#     return np.array(X), y

def get_mesh_data(SUBs, window_len=10, step=5, labels=[0,1,2,3,4]):
    EEG = eeg_import(window_len=window_len, step=step)
    X, y = EEG.get_data(SUBs)
    X = transform(X, window_len)
    lb = LabelBinarizer()
    lb.fit(labels)
    y = lb.transform(y)
    return np.array(X), y

def interpolation(X):
    # TODO
    return

def transform(X, window_len=10, interpolate=False):
    if interpolate:
        X = interpolation(X)
#     X = scale_data(X)
    X = [get_mesh_sequence(x, window_len) for x in X]
    return X
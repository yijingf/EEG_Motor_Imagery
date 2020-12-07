import os
import random
import numpy as np
from mne import pick_types
from mne.io import read_raw_edf
from collections import Counter
from sklearn.preprocessing import LabelBinarizer

from config import *

labels = [0,1,2,3,4]
lb = LabelBinarizer()
lb.fit(labels)
    
def read_file(fname, l_freq=1, h_freq=30, resample_sfreq=None):
    """
    Load EEG signal and events, and apply bandpass filter.
    Arguments:
        fname: str, absolute path
        l_freq: int
        h_freq: int 
        sfreq: int, resample frequency
    """
    raw = read_raw_edf(fname, preload=True, verbose=False)
    picks = pick_types(raw.info, eeg=True)
    
    # Reject samples with different sampling rate. This will be deprecated when resampling is applied 
    if raw.info['sfreq'] != 160:
        print('{} is sampled at 128Hz, will be excluded.'.format(subj))
        return

    # bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
    if sfreq:
        raw = raw.resample(resample_sfreq)
        
    data = raw.get_data(picks=picks)
    
    # Get annotation
    events = raw.find_edf_events()
    return data, events

def get_window(data, events, run_type, window_len, step, mesh=True, t_cue=t_cue):
    # Filter out rest states
    if run_type:
        events = [event for event in events if event[-1] != 'T0']
        
    x, y = [], []
    for j, event in enumerate(events):
        label = label_run[str(run_type)][event[2]]

        start = int((float(event[0]) + t_cue) * sfreq)
        duration = int((float(event[1]) - t_cue) * sfreq)

        tmp_data = data[:, start: start + duration]
        n_seg = int((duration - window_len)/step) + 1
        
        if mesh:
            tmp_data = np.array([get_mesh(tmp_data[:,i]) for i in range(duration)])
            x += [tmp_data[i * step: i * step + window_len, :,:] for i in range(n_seg)]
        else:
            x += [tmp_data[:, i * step: i * step + window_len] for i in range(n_seg)]
            
        y += [label for i in range(n_seg)]
        
    return np.array(x), y 

def balance_sample(Y):
    label_cnt = Counter(Y)
    min_cnt = min(label_cnt.values())
    min_label = min(label_cnt, key=label_cnt.get)
    selected_index = []
    for i in [i for i in labels if i != min_label]:
        tmp_index = np.where(Y == i)[0].tolist()
        selected_index += random.sample(tmp_index, min_cnt)
    selected_index += np.where(Y == min_label)[0].tolist()
    
    return selected_index

def split(index, split_ratio):
    """
    Randomly split dataset and return the index of samples for train/valid/test
    """
    # Normalize split ratio
    split_ratio = np.array(split_ratio)/sum(split_ratio)

    random.shuffle(index)

    n_train, n_valid, n_test = (split_ratio * len(index)).astype(int)
    train_index = index[:n_train]
    valid_index = index[n_train:n_train + n_valid]
    test_index = index[n_train+n_valid:]
    return train_index, valid_index, test_index

def normalization(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean)/std
    return X

def reject_sample(X, Y, threshold=0.0001):
    num = len(X)
    artifact_index = set(np.where(abs(X) >= threshold)[0])
    clean_index = list(set(range(num)).difference(artifact_index))
    X = X[clean_index]
    Y = Y[clean_index]
    return X, Y

def get_mesh(x, interpolate=False):
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

def transform(X, window_len=10, interpolate=False):
    if interpolate:
        X = interpolation(X)
#     X = scale_data(X)
    X = [get_mesh_sequence(x, window_len) for x in X]
    return X

class DataLoader():
    """
    Load eeg data from .edf/.event files.
    """
    def __init__(self, window_len=10, overlap=0.5, **kwargs):
        self.step = int(window_len*(1 - overlap))
        self.window_len = window_len

    def load_data(self, SUBs, normalized=True, mesh=True, l_freq=4, h_freq=30, resample_sfreq=None):

        X, Y = [], []
        window_cnt = []
        for sub in SUBs:
            len_sub = []
            run_cnt = 0
            for run_type, runs in enumerate(total_runs):
                len_runs = []
                for run in runs:
                    fname = os.path.join(dataDir, sub, '{}R{}.edf'.format(sub, run))
                    data, events = read_file(fname, l_freq, h_freq, resample_sfreq)
                    x, y = get_window(data, events, run_type, self.window_len, self.step, mesh)
                    run_cnt += len(y)
                    len_runs.append(run_cnt)
                    X.append(x)
                    Y.append(y)
                len_sub += len_runs
            window_cnt.append(len_sub)
        X = np.concatenate(X)
        Y = np.concatenate(Y)

#         if normalized:
#             X = normalization(X)
        
#         if reject:
#             X, Y = reject_sample(X, Y, threshold=reject_threshold)
        return X, Y, window_cnt
        
    def load_train_val_test(self, SUBs, normalized=True, mesh=True, 
    						l_freq=4, h_freq=30, resample_sfreq=None, 
    						one_hot=True):
        
        X, Y, window_cnt = self.load_data(SUBs, normalized, mesh, l_freq, h_freq, resample_sfreq)
        selected_index = balance_sample(Y)

        train_index, valid_index, test_index = split(selected_index, split_ratio)
        if one_hot:
            Y = lb.transform(Y)
        train_set = (X[train_index], Y[train_index])
        valid_set = (X[valid_index], Y[valid_index])
    
        return train_set, valid_set, (test_index, np.array(window_cnt))

class TestDataLoader():
    """
    Loading data for within subject model testing given the index of windows.
    The order of subjects are required to be consistent with that when loading training data.
    """
    
    def __init__(self, window_cnt, window_len=10, overlap=0.5, **kwargs):
        self.window_len = window_len
        self.step = int(window_len * overlap)

        self.window_cnt = window_cnt
        n_window_per_sub = np.array(window_cnt)[:,-1]
        self.n_window_cum = np.cumsum(n_window_per_sub)
        self.n_window_per_sub = np.array([0] + list(self.n_window_cum))
        return
    
    def get_fname(self, index):
        tmp = index/self.n_window_cum
        tmp[np.where(tmp >= 1)] = 0
        sub_index = np.argmax(tmp)
        
        tmp_run = index - self.n_window_per_sub[sub_index]
        tmp = tmp_run/self.window_cnt[sub_index]

        tmp[np.where(tmp >= 1)] = 0
        run_index = np.argmax(tmp)
        window_index = tmp_run - ([0] + self.window_cnt[sub_index].tolist())[run_index]
        
        run, run_type = test_total_runs[run_index]
        
        return sub_index, run, run_type, window_index
    
    def load_data(self, SUBs, test_index, mesh=True, one_hot=True):
        input_list = {}
        # retrieve the subject/run of the window by its index
        for index in test_index:
            sub_index, run, run_type, window_index = self.get_fname(index)
            sub = SUBs[sub_index]
            fname = os.path.join(dataDir, sub, '{}R{}.edf'.format(sub, run))

            # Group by filename to save time reading files
            if fname in input_list:
                input_list[fname]['window_index'] = input_list[fname]['window_index'] + [window_index]
            else:
                input_list[fname] = {'run_type':run_type, 'window_index':[]}

        X, Y = [], []
        for fname, v in input_list.items():
            run_type, window_index = v['run_type'], v['window_index']
            data, events = read_file(fname)
            x, y = get_window(data, events, run_type, self.window_len, self.step, mesh)
            
            X.append(x[window_index])
            Y.append(np.array(y)[window_index])

        X = np.concatenate(X)
        Y = np.concatenate(Y)
        if one_hot:
            Y = lb.transform(Y)
        return X, Y
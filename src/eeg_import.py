import os
import random
import numpy as np
from mne import pick_types
from mne.io import read_raw_edf
from collections import Counter

# Label & Run
label_run = {'0':{'T0':0} , '1':{'T1':1, 'T2':2}, '2':{'T1':3, 'T2':4}}
total_runs = [['02'], ['04','08','12'], ['06','10','14']]

PATH = '../data'
SUBs = sorted(os.listdir(PATH))
remove_list = ['S088', 'S089', 'S092', 'S100']
for sub in remove_list:
    try:
        SUBs.remove(sub)
    except:
        pass
    
def read_data(fname, run_type):
    raw = read_raw_edf(fname, preload=True, verbose=False)
    picks = pick_types(raw.info, eeg=True)

    if raw.info['sfreq'] != 160:
        print('{} is sampled at 128Hz so will be excluded.'.format(subj))
        return

    # High-pass filtering
    raw.filter(l_freq=1, h_freq=None, picks=picks)
    data = raw.get_data(picks=picks)
    # Get annotation
    events = raw.find_edf_events()
    if run_type:
        events = [event for event in events if event[-1] != 'T0']
    return data, events

def balance_sample(X, Y):
    num = Counter(Y)[0]
    index_1 = list(np.where(Y != 0)[0])
    index_0 = list(np.where(Y == 0)[0])
    index_1 = random.sample(index_1, num*4)
    index = index_0 + index_1
    X = X[index]
    Y = Y[index]
    return X, Y

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

class eeg_import():
    
    def __init__(self,
                 sfreq=160,
                 window_len=10,
                 step=None,
                 overlap=0.5, 
                 t_cue=0.5):
        if step:
            self.step = step
        else:
            self.step = int(window_len*overlap)
        self.sfreq = sfreq
        self.window_len = window_len
        self.t_window = self.window_len/self.sfreq
        self.t_step = self.step/sfreq 
        self.t_cue = 0.5 # extract data from 0.5s after the cue

    def get_window(self, data, events, run_type):
        x = []
        y = []
        for event in events:
            start = int((float(event[0]) + self.t_cue) * self.sfreq)
            duration = float(event[1]) - self.t_cue
            label = label_run[str(run_type)][event[2]]

            n_segments = int((duration-self.t_window)/self.t_step) + 1
            x += [data[:, start + i * self.step: start + i * self.step + self.window_len]
                  for i in range(n_segments)]
            y += [label for i in range(n_segments)]

        return np.array(x), y

    def get_data(self, SUBs, 
                 PATH=PATH,
                 balance=True,
                 normalized=True,
                 reject=True,
                 reject_threshold=0.0001):
        X = []
        Y = []
        for sub in SUBs:
            for run_type, runs in enumerate(total_runs):
                for run in runs:
                    fname = os.path.join(PATH, sub, '{}R{}.edf'.format(sub, run))
                    data, events = read_data(fname, run_type)
                    x, y = self.get_window(data, events, run_type)
                    X.append(x)
                    Y.append(y)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        if reject:
            X, Y = reject_sample(X, Y, threshold=reject_threshold)
        if normalized:
            X = normalization(X)
        if balance:
            X, Y = balance_sample(X, Y)
        return X, Y
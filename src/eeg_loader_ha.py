import os
import random
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelBinarizer

from config import *
from align import hyperalignment
from eeg_loader import read_file, split, get_mesh

# SUBs = SUBs[:5]

n_task = 90
n_rest = 15
n_task_per_run = 15
t_seg = 4

window_len = 80
step = 40

labels = [0,1,2,3,4]
lb = LabelBinarizer()
lb.fit(labels)

def split_ha(ResultDir):
    train_index_r, valid_index_r, test_index_r = [], [], []
    train_index_t, valid_index_t, test_index_t = [], [], []
    for sub in SUBs:
        id0_t, id1_t, id2_t = split(list(range(n_task)), split_ratio)
        train_index_t += [id0_t]
        valid_index_t += [id1_t]
        test_index_t += [id2_t]
        id0_r, id1_r, id2_r = split(list(range(n_rest)), split_ratio)
        train_index_r += [id0_r]
        valid_index_r += [id1_r]
        test_index_r += [id2_r]
        
    # np.savez(os.path.join(ResultDir, 'valid_index'), task=valid_index_t, rest=valid_index_r)
    np.savez(os.path.join(ResultDir, 'test_index'), task=test_index_t, rest=test_index_r)
    return train_index_t, train_index_r, valid_index_t, valid_index_r

def get_block(data, events, run_type, t_cue=0.5):
    x = [data[int((float(i[0])+t_cue)*sfreq): int(float(i[0])*sfreq)+int(float(i[1])*sfreq)] for i in events]
    y = [label_run[str(run_type)][i[2]] for i in events]
    duration = [int((float(i[1])-t_cue)*sfreq) for i in events]

    return np.concatenate(x), y, duration

def get_window(x_block, y_block, d, window_len=80, step=40, mesh=True):
    x = []
    y = []
    for tmp_x, tmp_y, tmp_d in zip(*(x_block, y_block, d)):
        n_seg = int((tmp_d - window_len)/step) + 1
        if mesh:
            tmp_x = [get_mesh(tmp_x[i]) for i in range(tmp_d)]
            tmp_x = [np.array(tmp_x[i*step:i*step + window_len]) for i in range(n_seg)]
        else:
            tmp_x = [tmp_x[i*step:i*step + window_len].T for i in range(n_seg)]      

        x += tmp_x
        y += [tmp_y for _ in range(n_seg)]
        
    return x, y

def load_data_from_file(index_r, index_t):
    X = []
    Y = []
    D = []
    for i, sub in enumerate(SUBs):
        x, y, d = [], [], []
        fname = os.path.join(dataDir, sub, '{}R{}.edf'.format(sub, '02'))
        data, events = read_file(fname)

        n_events = len(index_r[i])
        events = [i for i in zip(*(t_seg*np.array(index_r[i]), 
                                   [t_seg for _ in range(n_events)],
                                   ['T0' for _ in range(n_events)]))]

    #     Load first run
        x_r, y_r, d_r = get_block(data.T, events, 0)

        x += [x_r]
        y += [y_r]
        d += [d_r]
        index = sorted(index_t[i])
        cnt = 0
        for run_type, runs in enumerate(total_runs[1:]):
            for run in runs:
                fname = os.path.join(dataDir, sub, '{}R{}.edf'.format(sub, run))
                data, events = read_file(fname)

                events = [event for event in events if event[-1] != 'T0']
                cnt += len(events)
                tmp_index = [j for j in index if j < cnt]
                if not tmp_index:
                    continue
            
                events = np.array(events)[np.array(tmp_index) - cnt]
                index = index[len(tmp_index):]

                x_t, y_t, d_t = get_block(data.T, events, run_type+1)
                x += [x_t]
                y += [y_t]
                d += [d_t]

        X += [np.concatenate(x)]
        Y += [np.concatenate(y)]
        D += [np.concatenate(d)]
    return X, Y, D

def get_data(X, Y, D):
    new_X = []
    new_Y = []
    for i in range(len(SUBs)):
        d = D[i]
        x_block = X[i]
        y_block = Y[i]

        d_block = list(np.cumsum(d))
        x_block = [x_block[v:d_block[i]] for i, v in enumerate([0] + d_block[:-1])]
    
        x, y = get_window(x_block, y_block, d)
        new_X += x
        new_Y += y

    new_X = np.concatenate(new_X)
    new_Y = lb.transform(np.concatenate(new_Y))

    return new_X, new_Y 
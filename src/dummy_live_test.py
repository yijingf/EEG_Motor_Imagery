import collections
import time, os
import numpy as np
import tensorflow as tf
from EEGTest import *
from emokit.emotiv import Emotiv

# EEG info
sfreq = 128
window_len = 10
sliding_step = 5

# EEG Sliding Window
w, h = 10, 11

# Vote Window
vote_window_len = 10
vote_sliding_step = 5

# Initialize Model
eeg_test = EEG_Test(gpu_memory_fraction=0.5)
print('Warm-up Model')

# Run dummy data to warm up the model
window_data = np.zeros((window_len, w, h, 1))+1
window_label = eeg_test.predict([window_data])
print('Model Initialized')

read_interval = 0.001

num_channel = 14
s_range = 3
read_interval = 1/sfreq

vote_window = [2 for i in range(vote_window_len)]

while True:
    tmp_vote = []
    for i in range(vote_sliding_step):
        tmp_data = []
        for j in range(sliding_step):
            value = list(np.random.randint(s_range, size=(num_channel,) ))
            time.sleep(read_interval)
            tmp_data.append(value)
        tmp_data = trans_2d(tmp_data)
        window_data = np.concatenate((window_data[window_len-sliding_step:], 
                                      tmp_data), axis = 0)
        window_label = eeg_test.predict([window_data])[0]
        tmp_vote.append(window_label)
    vote_window = vote_window[vote_window_len-vote_sliding_step:] + tmp_vote
    res = voting(list(vote_window))
    print('res', res)
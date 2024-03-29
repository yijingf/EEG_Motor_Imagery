dataDir = '../data'
resPath = '../res'

# Sequence Config
t_cue = 0.5 # extract data from 0.5s after the cue
sfreq = 160 # Sampling rate

# Train Valid Test
split_ratio = [0.6, 0.2, 0.2]

# Frequency bands for bandpower extraction
freq_bands = [[4,7], [8,13], [13, 30]] # theta, alpha, beta

# Label & Run
labels = [0, 1, 2, 3, 4] # Todo specify catefory
label_run = {'0':{'T0':0} , '1':{'T1':1, 'T2':2}, '2':{'T1':3, 'T2':4}}
total_runs = [['02'], ['04','08','12'], ['06','10','14']]
test_total_runs = [[j, k] for k, i in enumerate(total_runs) for j in i]

# Data Input Config
with open('../config/subs.txt', 'r') as f:
    SUBs = f.read().splitlines()

with open('../config/train_subs.txt', 'r') as f:
    trainSUBs = f.read().splitlines()
    
with open('../config/valid_subs.txt', 'r') as f:
    validSUBs = f.read().splitlines()
        
with open('../config/test_subs.txt', 'r') as f:
    testSUBs = f.read().splitlines()
dataDir = '../data'

# Sequence Config
t_cue = 0.5 # extract data from 0.5s after the cue
sfreq = 160 # Sampling rate

# Train Valid Test
split_ratio = [0.6, 0.2, 0.2]

# Label & Run
labels = [0, 1, 2, 3, 4] # Todo specify catefory
label_run = {'0':{'T0':0} , '1':{'T1':1, 'T2':2}, '2':{'T1':3, 'T2':4}}
total_runs = [['02'], ['04','08','12'], ['06','10','14']]
test_total_runs = [[j, k] for k, i in enumerate(total_runs) for j in i]
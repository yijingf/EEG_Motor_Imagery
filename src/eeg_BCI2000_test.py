import time, os, sys, json

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

from eeg_loader import *
from config import *

# EEG info
sfreq = 160

class EEG_Test:
    
    def __init__(self, gpu_usage=True, gpu_memory_fraction=0.1, 
                 model_dir='./model', model='model.ckpt'):
        
        # GPU Configuration
        if not gpu_usage:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            self.config = tf.ConfigProto()
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.config = tf.ConfigProto(gpu_options=gpu_options)
            
        self.load_model(model_dir, model)
        
    def warm_up(self):
        # To Do
        # warm up the network before testing will speed up the process
        return
    
    def load_model(self, model_dir, model):
        self.sess = tf.Session(config=self.config)
        saver = tf.train.import_meta_graph(os.path.join(model_dir, model+'.meta'))
        saver.restore(self.sess, os.path.join(model_dir, model))
        graph = tf.get_default_graph()
        
        self.x = graph.get_operation_by_name('x_input').outputs[0]
        
        self.is_training = graph.get_operation_by_name('is_training').outputs[0]
#         self.seqlen = graph.get_operation_by_name('seqlen').outputs[0]
        self.prediction = graph.get_operation_by_name('prediction').outputs[0]
        return
    
    def predict(self, x_input):
        feed_dict = {self.x: x_input, self.is_training: False}
        res = self.sess.run(self.prediction, feed_dict=feed_dict)
        return res
    
def voting(index):
    return int(Counter(index).most_common(1)[0][0])

def demo_simulate(sub_id, model_dir, model, 
                  window_len=10, vote_window_len=10, vote_sliding_step=5, **kwargs):

    windowTime = window_len/sfreq
    vote_window_time = windowTime * vote_window_len
    
    interval = vote_sliding_step * window_len/sfreq
    
    # Load Model
    print('Loading Model')
    eeg_test = EEG_Test(gpu_memory_fraction=0.1, model_dir=model_dir, model=model)
    
    # Load Data
    print('Loading Data')
    sub = 'S{:03d}'.format(sub_id)
    X, y = get_mesh_data([sub])
    
    # Simulation
    total = len(y)
    vote_res = [0 for i in range(vote_window_len)]
    for i in range(0, total, vote_sliding_step):
        x_input = X[i:i+vote_sliding_step]
        res = list(eeg_test.predict(x_input))
        vote_res = vote_res[vote_window_len-vote_sliding_step:] + res
        res = voting(vote_res)
        print('Predicted Label: {}; Ground Truth: NA'.format(res))
        time.sleep(interval)
        
def test(model_dir, model, batch_size=128, **kwargs):
    batch_size = 128

    # Load Model
    print('Loading Model')
    eeg_test = EEG_Test(gpu_memory_fraction=0.1, model_dir=model_dir, model=model)

    # Load Data
    if input_config['mode'] == 'within':
        test_set = np.load(os.path.join(res_dir, 'test_set'))
        data_loader = TestDataLoader(window_cnt=test_set['window_cnt'], **input_config)
        X, y = data_loader.load_data(SUBs, test_set['test_index'])
    else:
        data_loader = DataLoader(**input_config)
        X, y, _ = data_loader.load_data(testSUBs)

    n = len(y)
    index = list(range(n))
    data_split = np.array_split(index, n//batch_size)

    y_pred = []
    for i, ds in enumerate(data_split):
        tmp_y_pred = eeg_test.predict(X[ds])
        y_pred += tmp_y_pred.tolist()

    y = np.argmax(y, axis=1).tolist()
    output_filename = os.path.join(res_dir, 'predicted_label_test.csv')
    res = pd.DataFrame({'y_true':y, 'y_pred':y_pred})
    res.to_csv(output_filename, index=False)
    return    
    
def voting(index):
    return int(collections.Counter(index).most_common(1)[0][0])

def demo_simulate(sub_id, model_dir, model, 
                  window_len=10, vote_window_len=10, vote_sliding_step=5, **kwargs):

    windowTime = window_len/sfreq
    vote_window_time = windowTime * vote_window_len
    
    interval = vote_sliding_step * window_len/sfreq
    
    # Load Model
    print('Loading Model')
    eeg_test = EEG_Test(gpu_memory_fraction=0.1, model_dir=model_dir, model=model)
    
    # Load Data
    print('Loading Data')
    sub = 'S{:03d}'.format(sub_id)
    X, y = get_mesh_data([sub])
    
    # Simulation
    total = len(y)
    vote_res = [0 for i in range(vote_window_len)]
    for i in range(0, total, vote_sliding_step):
        x_input = X[i:i+vote_sliding_step]
        res = list(eeg_test.predict(x_input))
        vote_res = vote_res[vote_window_len-vote_sliding_step:] + res
        res = voting(vote_res)
        print('Predicted Label: {}; Ground Truth: NA'.format(res))
        time.sleep(interval)


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--mode', dest='mode', type=str, help='Test mode (i for individual, b for batch)')
    parser.add_argument('-i', '--id', dest="sub_id", type=str, help="Test Subject ID")
    parser.add_argument('-p', '--prefix', dest='prefix', type=str)
    
    args = parser.parse_args()
    
    if args.prefix:
        prefix = args.prefix
    else:
        sys.exit('Please provide model index, usually date in the form of yymmddhh')
        
    with open('../result/{}/model_conf.json'.format(prefix), 'r') as f:
        model_conf = json.load(f)
        
    model_dir = '../model/{}'.format(prefix)
    model = 'model.ckpt' # or 'batch_model.ckpt'
    
    mode = args.mode or 'b'

    res_dir = '../result/{}'.format(prefix)
    
    print(mode, model)

    input_config = model_conf['input']
    
    if mode == 'i':
        sub_id = args.sub_id or 1
        demo_simulate(sub_id, model_dir, **model_conf['input'])
    elif mode == 'b':
        with open('../config/test_subs.txt', 'r') as f:
            test_subs = f.read().splitlines()
        test(model_dir, model)
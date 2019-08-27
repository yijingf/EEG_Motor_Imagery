import collections
import time, os, sys
import numpy as np
import tensorflow as tf
from import_data import get_mesh_data

# EEG info
sfreq = 160
window_len = 10
windowTime = window_len/sfreq

vote_window_len = 10
vote_sliding_step = 5
vote_window_time = windowTime * vote_window_len
interval = vote_sliding_step * window_len/sfreq


class EEG_Test:
    
    def __init__(self, 
                 gpu_usage=True,
                 gpu_memory_fraction=0.1, 
                 model_dir='./model',
                 model='model.ckpt'):
        
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
        self.seqlen = graph.get_operation_by_name('seqlen').outputs[0]
        self.prediction = graph.get_operation_by_name('prediction').outputs[0]
        return
    
    def predict(self, x_input):
        batch_size = x_input.shape[0]
        batch_seqlen = [window_len for i in range(batch_size)]
        feed_dict = {self.x: x_input, 
                     self.is_training: False, 
                     self.seqlen: batch_seqlen}
        res = self.sess.run(self.prediction, feed_dict=feed_dict)
        return res

def voting(index):
    return int(collections.Counter(index).most_common(1)[0][0])

def demo_simulate(sub_id):
    # Load Model
    print('Loading Model')
    eeg_test = EEG_Test(gpu_memory_fraction=0.1, model_dir='./model/')
    
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
    parser.add_argument('-i', '--id', dest="sub_id", type = str,
                        help="Test Subject ID")

    args = parser.parse_args()
    
    if not args.sub_id:
        sub_id = 1
    sub_id = args.sub_id
    demo_simulate(sub_id)
import os, time, json
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

from layers import *
from Net import *
from import_data import get_mesh_data

from CasCNNRNN import *

date = datetime.now().strftime("%Y%m%d")

ModelDir = '../{}_model'.format(date)
os.makedirs(ModelDir, exist_ok=True)

ModelFile = os.path.join(ModelDir, 'model.ckpt')
Batch_ModelFile = os.path.join(ModelDir, 'batch_model.ckpt')
ResultDir = '../result/{}'.format(date)
os.makedirs(ResultDir, exist_ok=True)

PATH = '../data/'

with open('../config/train_subs.txt', 'r') as f:
    SUBs = f.read().splitlines()

# Initialize Input Parameter (fixed)
w, h = 10, 11
input_channel = 1
n_classes = 5

class EEG_Train:
    """
    Randomly leave out 21 subjects, and train and validate the model on the rest subjects.
    Arguments:
        leave_out_index: `int`.
        gpu_memory_fraction: `float`
    """
    def __init__(self,
                 gpu_usage=True,
                 gpu_memory_fraction=0.5):
        
        self.SUBs = SUBs
        
        # GPU-Configuration
        if not gpu_usage:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            self.config = tf.ConfigProto()
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.config = tf.ConfigProto(gpu_options=gpu_options)
        
    def load_data(self, valid_ratio=0.2):
        print('Loading Data')
        X, Y = get_mesh_data(self.SUBs, window_len=window_len, step=step)
        
        total_num = len(Y)
        num_train, num_valid = int(round(total_num * (1 - valid_ratio))), int(round(total_num * valid_ratio))
        total_index = np.array(list(range(total_num)))
        random.shuffle(total_index)

        train_x, train_y = X[total_index[:num_train]], Y[total_index[:num_train]]
        valid_x, valid_y = X[total_index[num_train:]], Y[total_index[num_train:]]
        del X
        del Y
        print('Data Loaded')
        return train_x, train_y, valid_x, valid_y
        
    def train(self, model_conf, resume=False):
        
        sess = tf.Session(config = self.config)
    
        if resume:
            print("Loading Model")
            saver = tf.train.import_meta_graph(os.path.join(ModelDir, ModelFile+'.meta'))
            saver.restore(sess, os.path.join(ModelDir, ModelFile))
            graph = tf.get_default_graph()
            x = graph.get_operation_by_name('x_input').outputs[0]
            y = graph.get_operation_by_name('y_input').outputs[0]
            is_training = graph.get_operation_by_name('is_training').outputs[0]
            loss_op = graph.get_operation_by_name('loss_op').outputs[0]
            train_op = graph.get_operation_by_name('train_op')
            predicted = graph.get_operation_by_name('prediction').outputs[0]
           
        else:
            # Initialize input variable
            x = tf.placeholder(tf.float32,
                               [None, window_len, w, h, input_channel],
                               name = 'x_input')
            y = tf.placeholder(tf.float32, [None, n_classes], name = 'y_input')
            is_training = tf.placeholder(tf.bool, name = 'is_training')

            net = tf.unstack(x, window_len, 1)

            cnn_type = model_conf['cnn']['net_type']
            if cnn_type == 'ResNet':
                net = [CasRes(tmp, is_training, **model_conf['cnn']) for tmp in net]
            elif cnn_type == 'CNN':
                net = [CasCNN(tmp, **model_conf['cnn']) for tmp in net]

            logits = CasRNN(net, **model_conf['rnn'])

            regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels = y))
            if regularization:
                regularization = tf.add_n(regularization, name='Regularization')
                loss_op = tf.add(loss_op, regularization, name='loss_op')
            else:
                loss_op = tf.identity(loss_op, name="loss_op")

            # Optimization    
            # get moving average in batch normalization layers
            op_collection = 'batch_normalization'
            bn_op = tf.get_collection(op_collection)

            # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = tf.group(optimizer.minimize(loss_op), *bn_op, name = 'train_op')

            predicted = tf.argmax(logits, 1, name='prediction')
            
        correct_pred = tf.equal(predicted, tf.argmax(y, 1))
        num_correct = tf.count_nonzero(correct_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        if not resume:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init, {is_training: True})
            
        train_x, train_y, valid_x, valid_y = self.load_data()
        num_train, num_valid = len(train_y), len(valid_y)
        train_index, valid_index = list(range(num_train)), list(range(num_valid))
        
        # Training
        train_filename = os.path.join(ResultDir, 'train_log.txt')
        valid_filename = os.path.join(ResultDir, 'val_log.txt')

        best_acc = 0
        best_batch_acc = 0
        step_count = 0

        for e in range(num_epoch):
            random.shuffle(train_index)
            train_split = np.array_split(train_index, num_train//batch_size)
            valid_split = np.array_split(valid_index, num_valid//batch_size)    

            """
            Training
            """
            batch_count = len(train_split)
            t_start = time.time()
            for i, ts in enumerate(train_split):
                x_img, y_label = train_x[ts], train_y[ts]
                feed_dict={x: x_img, y: y_label, is_training: True}

                # Display and save training accuracy per batch
                if step_count % display_step == 0:
                    _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict=feed_dict)
                    display = 'Step: {}, Loss: {:.6f}, Acc: {:.6f}'.format(step_count, loss, acc)
                    print(display, end='\r')
                    with open(train_filename, 'a') as f:
                        f.write(display + '\n')
                else:
                    sess.run(train_op, feed_dict=feed_dict)

                # Display validation accuracy on smaller validation dataset
                if step_count % num_valid_step == 0:
                    vs = valid_split[int(i/display_step)]
                    valid_dict = {x: valid_x[vs], y: valid_y[vs], is_training: False}
                    loss, acc = sess.run([loss_op, accuracy], feed_dict=valid_dict)
                    display = 'Validation Step: {}, Loss: {:.6f}, Acc: {:.6f}'.format(step_count, loss, acc)
                    print(display)
                    if acc >= best_batch_acc:
                        best_batch_acc = acc
                        save_path = saver.save(sess, Batch_ModelFile)
                        print("Model saved in path: %s" % save_path)
                    with open(valid_filename, 'a') as f:
                        f.write(display + '\n')
                    
                step_count += 1

            t_end = int((time.time() - t_start)/60)
            print("Time for this epoch: {} min".format(t_end))

            """
            Validation
            """
            val_total_correct = 0
            for vs in valid_split:
                feed_dict={x: valid_x[vs], y: valid_y[vs], is_training: False}
                nc = sess.run(num_correct, feed_dict=feed_dict)
                val_total_correct += nc

            acc = val_total_correct/num_valid
            display = 'Epch: %d\tVal acc: %.8f'%(e, acc)
            with open(valid_filename, 'a') as f:
                f.write(display + '\n')
            print(display)

            if acc >= best_acc:
                best_acc = acc
                save_path = saver.save(sess, ModelFile)
                print("Model saved in path: %s" % save_path)
                with open(valid_filename, 'a') as f:
                    f.write("Model saved" + '\n')
        return
    
if __name__ == '__main__':

    # Loading Model Configuration
    model_conf_filename=os.path.join(ResultDir, 'model_conf.json')
    with open(model_conf_filename, 'r') as f:
        model_conf = json.load(f)

    train_conf_filename=os.path.join(ResultDir, 'train_conf.json')
    with open(train_conf_filename, 'r') as f:
        train_conf = json.load(f)

    # Hyper Parameter Initialization

    window_len = model_conf['input']['window_len']
    step = model_conf['input']['step']
    learning_rate = train_conf['learning_rate']
    num_epoch = train_conf['num_epoch']
    batch_size = train_conf['batch_size']
    display_step = train_conf['display_step']
    num_valid_step = train_conf['num_valid_step']

    eeg_train = EEG_Train(gpu_memory_fraction=0.85)
    res = eeg_train.train(model_conf)
    # eeg_train = EEG_Train()
    # res = eeg_train.train()

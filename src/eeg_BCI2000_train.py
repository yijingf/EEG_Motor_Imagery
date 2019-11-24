import os, time, json
import random
import numpy as np
import tensorflow as tf

from CasCNNRNN import *
from eeg_loader import *

from train_config import *

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
    def __init__(self, gpu_usage=True, gpu_memory_fraction=0.5):
        
        # GPU-Configuration
        if not gpu_usage:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
            self.config = tf.ConfigProto()
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            self.config = tf.ConfigProto(gpu_options=gpu_options)

    def load_data(self):
        data_loader = DataLoader(**input_config)

        if input_config['mode'] == 'within':
            (train_X, train_y), (valid_X, valid_y), test_set = data_loader.load_train_val_test(SUBs)

            # Save test index
            np.savez(test_index_filename, test_index=test_set[0], window_cnt=test_set[1])
        else:
            train_X, train_y, _ = data_loader.load_data(trainSUBs)
            valid_X, valid_y, _ = data_loader.load_data(validSUBs)

            index = balance_sample(train_y)
            train_X, train_y = train_X[index], train_y[index]

            valid_index = balance_sample(valid_y)
            valid_X, valid_y = valid_X[index], valid_y[index]
            
        return train_X, train_y, valid_X, valid_y

    def train(self, resume=False):
        
        sess = tf.Session(config = self.config)
    
        if resume:
            print("Loading Model")
            saver = tf.train.import_meta_graph(Batch_ModelFile+'.meta')
            saver.restore(sess, Batch_ModelFile)
            graph = tf.get_default_graph()
            x_cnn = graph.get_operation_by_name('x_input').outputs[0]
            x_rnn = [graph.get_operation_by_name('x_rnn_input_{}'.format(i)).outputs[0] for i in range(window_len)]
            y = graph.get_operation_by_name('y_input').outputs[0]
            is_training = graph.get_operation_by_name('is_training').outputs[0]
            loss_op = graph.get_operation_by_name('loss_op').outputs[0]
            train_op = graph.get_operation_by_name('train_op')
            predicted = graph.get_operation_by_name('prediction').outputs[0]
           
        else:
            # Initialize input variable
            x_cnn = tf.placeholder(tf.float32, [None, w, h, input_channel], name='x_input')
            rnn_dimension = cnn_config['fc_n_hidden']
            x_rnn = [tf.placeholder(tf.float32, [None, rnn_dimension], name='x_rnn_input') for _ in range(window_len)]
            y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')
            is_training = tf.placeholder(tf.bool, name='is_training')

            cnn_type = cnn_config['net_type']
            if cnn_type == 'ResNet':
                cnn_output = CasRes(x_cnn, is_training, **cnn_config)
            elif cnn_type == 'CNN':
                cnn_output = CasCNN(tmp, **cnn_config)
            
            logits = CasRNN(x_rnn, **rnn_config)

            regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
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
        
        # Load Data
        train_X, train_y, valid_X, valid_y = self.load_data()
        num_valid = len(valid_y)

        n_train, n_valid = len(train_y), len(valid_y)
        train_index, valid_index = list(range(n_train)), list(range(n_valid))

        # Training
        train_filename = os.path.join(ResultDir, 'train_log.txt')
        valid_filename = os.path.join(ResultDir, 'val_log.txt')
            
        if resume:
            Resume_message = '############## Resume Training ##############' 
            with open(train_filename, 'a') as f:
                f.write(Resume_message + '\n')
            with open(valid_filename, 'a') as f:
                f.write(Resume_message + '\n')
                
        best_acc = 0
        best_batch_acc = 0
        step_count = 0

        def feed_data(x_input, y_label, train_mode=True):
            # Flatten input sequence
            flatten_x_input = [j for i in x_input for j in i]


            # Feed data to cnn
            cnn_feed_dict = {x_cnn: flatten_x_input, is_training: train_mode}
            x_rnn_input = sess.run(cnn_output, feed_dict=cnn_feed_dict)

            # Feed data to rnn
            x_rnn_input = x_rnn_input.reshape((len(y_label), window_len, rnn_dimension))
            feed_dict = {x_rnn[i]:x_rnn_input[:, i, :] for i in range(window_len)}
            feed_dict[y] = y_label

            feed_dict.update(cnn_feed_dict)

            return feed_dict

        for e in range(num_epoch):
            random.shuffle(train_index)
            train_split = np.array_split(train_index, n_train//batch_size)
            valid_split = np.array_split(valid_index, n_valid//batch_size)    

            """
            Training
            """
            batch_count = len(train_split)
            t_start = time.time()
            for i, ts in enumerate(train_split):
                feed_dict = feed_data(train_X[ts], train_y[ts], train_mode=True)

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
                    valid_dict = feed_data(valid_X[vs], valid_y[vs], train_mode=False)
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
                feed_dict = feed_data(valid_X[vs], valid_y[vs], train_mode=False)

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
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode', type=str, help='r for resume')
    
    args = parser.parse_args()
    
    if args.mode == 'r':
        resume=True
    else:
        resume=False

    # Todo: Load Configuraion for Resume Mode

    # Hyper Parameter Initialization
    window_len = input_config['window_len']
    learning_rate = train_config['learning_rate']
    num_epoch = train_config['num_epoch']
    batch_size = train_config['batch_size']
    display_step = train_config['display_step']
    num_valid_step = train_config['num_valid_step']

    eeg_train = EEG_Train(gpu_memory_fraction=0.85)
    res = eeg_train.train(resume)
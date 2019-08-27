import os, time
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

from layers import *
from Net import *
from eeg_import import get_data
from eeg_preprocessing import prepare_data
# from import_data import get_mesh_data

from CasCNNRNN import *

# ModelDir = '/rigel/pimri/users/xh2170/data2/model_resnet'
# ModelFile = os.path.join(ModelDir, 'model/model.ckpt')
# ResultDir = '/rigel/pimri/users/xh2170/data2/model_resnet/result'
ModelDir = '/home/yf2375/ResNet_EEG/model'
ModelFile = os.path.join(ModelDir, 'model/model.ckpt')
ResultDir = '/home/yf2375/ResNet_EEG/result'

# Configuration
gpu_memory_fraction = 0.8
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
config = tf.ConfigProto(
    gpu_options=gpu_options
)

# Initialize Parameter
window_len = 10
w, h = 10, 11
input_channel = 1
n_hidden = 64
n_classes = 5

# Training Parameter
learning_rate = 0.01
momentum = 0.9
num_epoch = 10
training_steps = num_epoch * 35000
batch_size = 64

# Initialize Variable
x = tf.placeholder(tf.float32,
                   [None, window_len, w, h, input_channel],
                   name = 'x_input')
y = tf.placeholder(tf.float32, [None, n_classes], name = 'y_input')

is_training = tf.placeholder(tf.bool, name = 'is_training')
seqlen = tf.placeholder(tf.int32, [None], name = 'seqlen')

net = tf.unstack(x, window_len, 1)
net = [CasCNN(tmp, is_training) for tmp in net]
logits = CasRNN(net, seqlen)

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

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
train_op = tf.group(optimizer.minimize(loss_op), *bn_op, name = 'train_op')

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
predicted = tf.argmax(logits, 1)
num_correct = tf.count_nonzero(correct_pred)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print('Loading Data')

# Load Data
X, y = get_data()
train_x, train_y, valid_x, valid_y = prepare_data(X, y, test_ratio=0.25)
del X
del y
num_train, num_valid = len(train_x), len(train_y)
train_index, valid_index = list(range(num_train)), list(range(num_valid))

print('Data Loaded')
 
# X, Y = get_mesh_data()

# total_num = len(Y)
# valid_ratio = 0.25
# num_train, num_valid = int(round(total_num * (1 - valid_ratio))), int(round(total_num * valid_ratio))
# total_index = np.array(list(range(total_num))
# random.shuffle(total_index)

# train_x, train_y = X[:num_train], Y[:num_train]
# valid_x, valid_y = X[num_train:], Y[num_train:]

# train_index, valid_index = list(range(num_train)), list(range(num_valid))

# Training
date = datetime.now().strftime("%Y%m%d")
train_filename = 'train_log_{}.txt'.format(date[2:])
train_filename = os.path.join(ResultDir, train_filename)
valid_filename = 'val_log_{}.txt'.format(date[2:])
valid_filename = os.path.join(ResultDir, valid_filename)

init = tf.global_variables_initializer()

best_acc = 0
step_count = 0
with tf.Session(config = config) as sess:
    sess.run(init, {is_training: True})
    saver = tf.train.Saver()
    
#     if write_summary:
#         train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
#         validation_writer = tf.summary.FileWriter(summary_dir + '/validation', sess.graph)
    
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
            step_count += 1
            x_img, y_label = train_x[ts], train_y[ts]
            feed_dict={x: x_img, y: y_label, is_training: True}
            if step_count % display_step == 0:
                _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict=feed_dict)
                display = 'Step: {}, Loss: {:.6f}, Acc: {:.6f}'.format(step_count, loss, acc)
                with open(train_filename, 'a') as f:
                    f.write(display + '\n')
                print(display)
                if write_summary:
                    summary = sess.run(merged, feed_dict = feed_dict)
                    train_writer.add_summary(summary, e * batch_count + i)
            else:
                sess.run(train_op, feed_dict=feed_dict)
            if step_count == training_steps:
                break
            t_end = int((time.time()/60))
            
            print("Time for this epoch: {} min".format(t_end))
        """
        Validation
        """
        val_total_correct = 0
        for vs in valid_split:
            x_img, y_label = valid_x[vs], valid_y[vs]
            feed_dict={x: x_img, y: y_label, is_training: False}
            nc = sess.run(num_correct, feed_dict=feed_dict)
            val_total_correct += nc
            
        acc = float(val_total_correct)/num_valid
        
        display = 'Epch: %d\tVal acc: %.8f'%(e, acc)
        with open(valid_filename, 'a') as f:
            f.write(display + '\n')
        print(display)
        
#         if write_summary:
#             summary_v.value.add(tag='val_accuracy', simple_value = acc) 
#             validation_writer.add_summary(summary_v, e)
#             train_writer.flush()
#             validation_writer.flush()
        
        if acc >= best_acc:
            best_acc = acc
            save_path = saver.save(sess, ModelFile)
            print("Model saved in path: %s" % save_path)
            with open(valid_filename, 'a') as f:
                f.write("Model saved" + '\n')

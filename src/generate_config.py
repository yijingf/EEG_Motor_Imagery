import os, json
from datetime import datetime

# Model Config
input_config = dict(window_len=20,
                    step=10)

cnn_config = dict(net_type='CNN',
                  reuse=True,
                  conv_filter_shape = [{'W': [3,3,1,32], 'b': [32]}, 
                             {'W': [3,3,32,64], 'b': [64]},
                             {'W': [3,3,64,128], 'b': [128]}],
                  weight_decay=0.0005,
                  fc_layer=True,
                  fc_n_hidden=1024)

# cnn_config = dict(net_type='ResNet',
#                   reuse=True,
#                   conv_filter_shape=[{'W': [3,3,1,16], 'b': [16]}],
#                   n_blocks = [2, 1, 2, 1, 2],
#                   downsample = [False, True, False, True, False],
#                   resnet_filter_shape = [{'W0': [3,3,16,16], 'b0': [16]},
#                                          {'W0': [3,3,16,32], 'b0': [32], 'W1': [3,3,32,32], 'b1': [32]},
#                                          {'W0': [3,3,32,32], 'b0': [32]},
#                                          {'W0': [3,3,32,64], 'b0': [64], 'W1': [3,3,64,64], 'b1': [64]}, 
#                                          {'W0': [3,3,64,64], 'b0': [64]}
#                                         ],
#                   bias=False,
#                   weight_decay=0.0005,
#                   fc_layer=True,
#                   fc_n_hidden=512)

# rnn_config = dict(n_hidden=[64,16],
#                   bidirect=True,
#                   weight_decay=0.0001)

rnn_config = dict(n_hidden=[64],
                  bidirect=True,
                  weight_decay=0.0001)
    
# Training Hyper Parameter
params = dict(learning_rate=0.0001,
              num_epoch = 500,
              batch_size = 128,
              display_step = 8, 
              num_valid_step = 200)

date = datetime.today().strftime("%Y%m%d")

ResultDir = '../result/{}'.format(date)
os.makedirs(ResultDir, exist_ok=True)

param_filename = os.path.join(ResultDir, 'train_conf.json'.format(date[2:]))
with open(param_filename, 'w') as f:
    json.dump(params, f)
    
model_conf = {'input':input_config, 'cnn':cnn_config, 'rnn':rnn_config}
model_conf_filename = os.path.join(ResultDir, 'model_conf.json')
with open(model_conf_filename, 'w') as f:
    json.dump(model_conf, f)
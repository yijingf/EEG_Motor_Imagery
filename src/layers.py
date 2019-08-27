import tensorflow as tf
from tensorflow.python.training import moving_averages

# Parameter Initialization
def initialization(name, 
                   init_value=None, shape=None, initializer=None,
                   regularizer=None, weight_decay=0.01,
                   scope = None, trainable = True):
    """
    Arguments:
        name: `str`, variable name
        init_value: `numpy array`, Initial value of the parameter.
        shape: `list`, shape of tensor
        initializer: `object`
        regularizer: `str` L1
        scope: `str`
        trainable: `bool`
    """
    if regularizer == 'L2':
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

    if init_value is None:
        param = tf.get_variable(name, 
                                initializer=initializer, shape=shape,
                                regularizer=regularizer,
                                trainable=trainable)
    else:
        param = tf.get_variable(name,
                                initializer=tf.constant(init_value), 
                                regularizer=regularizer,
                                trainable=trainable)
    return param

# Convolutional Layer
def conv2d(x,
           W_shape=None, W_init=None, init_mean=0.0, init_std=0.02,
           b_shape=None, b_init=None, 
           regularizer=None, weight_decay=0.01,
           strides=1, 
           padding='SAME',
           name='Conv2D',
           scope=None,
           reuse=False, bias=True):
    """
    Arguments:
        x: input network;
        W_init/b_init: `numpy array`, initial value of W/b; 
        W_shape/b_shape: `list`, shape of W/b;
        W_initializer/b_initialier: `object` tensorflow initializer;
        regualarizer: `str`;
        weight_decay: `float`, regularization parameter;
        strides: `int`;
    """
    W_initializer = tf.random_normal_initializer(mean=init_mean, stddev=init_std)
    if bias:
        b_initializer = tf.random_uniform_initializer
    
    with tf.variable_scope(scope, default_name=name, reuse=reuse) as scope:
        
        W = initialization('W',
                           init_value=W_init, shape=W_shape, initializer=W_initializer, 
                           regularizer=regularizer, weight_decay=weight_decay,
                           trainable = True)
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)

        if bias:
            b = initialization('b', 
                               init_value=b_init, shape=b_shape, initializer=b_initializer, 
                               trainable=True)
            x = tf.nn.bias_add(x, b)
    return x

# Residual Block
def param_init(n_blocks):
    return [None for i in range(2 * n_blocks)]

def residual_block(is_training, x, n_blocks = 1, 
                   all_strides = None, 
                   W_shape_0 = None, W_init = None, W_shape_1 = None,
                   b_shape_0 = None, b_init = None, b_shape_1 = None,
                   regularizer=None, weight_decay=0.01,
                   padding = 'SAME',
                   beta = None, gamma = None,
                   batch_norm = True, relu = True, 
                   name = 'ResidualBlock', scope = None, 
                   reuse=False, bias=True):
    """
    Arguments:
        is_training: `tf.Variable`, training status
        x: input
        n_blocks: number of 2 layers of Conv2D
        W_shape/b_shape: `list`
        W_init/b_init: `numpy array` 
        beta, gamma: `list of numpy array`, initial value of beta/gamma
        batch_norm `bool`, batch normalization for every batch
        relu: `bool`, non-linear activation using relu
        name: `str`, name of the variable scope
    """
    input_shape = x.get_shape().as_list()
    in_n_channel = input_shape[-1]
                
    if W_init is None:
        W_init = param_init(n_blocks)
    if b_init is None:
        b_init = param_init(n_blocks)
    if all_strides is None:
        all_strides = [1,1]
    if beta is None:
        beta = param_init(n_blocks)
    if gamma is None:
        gamma = param_init(n_blocks)
    
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        
        for i in range(n_blocks):
            identity = x
        
            if batch_norm:

                batch_norm_name = 'BatchNormalization'
                if i*2 > 0:
                    batch_norm_name = batch_norm_name + '_{}'.format(i*2)
                x = batch_normalization(is_training, x, 
                                        beta=beta[i*2], gamma=gamma[i*2], 
                                        scope=batch_norm_name,
                                        reuse=reuse)

            x = tf.nn.relu(x)

            conv_name = 'Conv2D'
            if i*2 > 0:
                conv_name = conv_name + '_{}'.format(i*2)
                    
            x = conv2d(x,
                       W_init = W_init[i*2], W_shape = W_shape_0,
                       b_init = b_init[i*2], b_shape = b_shape_0, 
                       regularizer=regularizer, weight_decay=weight_decay,
                       strides=all_strides[0],
                       scope=conv_name,
                       reuse=reuse, bias=bias)

            if batch_norm:
                batch_norm_name = 'BatchNormalization_{}'.format(i*2+1)
                x = batch_normalization(is_training, x,
                                        beta=beta[i*2+1], gamma=gamma[i*2+1], 
                                        scope=batch_norm_name, 
                                        reuse=reuse)
            x = tf.nn.relu(x)
            
            if not W_shape_1:
                W_shape_1 = W_shape_0
            if not b_shape_1:
                b_shape_1 = b_shape_0
            
            conv_name = 'Conv2D_{}'.format(i*2+1)
            
            x = conv2d(x,
                       W_init = W_init[i*2+1], W_shape = W_shape_1,
                       b_init = b_init[i*2+1], b_shape = b_shape_1,
                       regularizer=regularizer, weight_decay=weight_decay,
                       strides = all_strides[1], 
                       scope=conv_name, 
                       reuse=reuse, bias=bias)

            # reshape identity if down-sampled
            if all_strides[0] > 1:
                strides = ksize = [1, all_strides[0], all_strides[0], 1]
                identity = tf.nn.avg_pool(identity, ksize=ksize, strides=strides, padding='SAME')

            # reshape identity (Zero-Padding)
            output_shape = x.get_shape().as_list()
            out_n_channel = output_shape[-1]
            if out_n_channel != in_n_channel:
                ch = (out_n_channel - in_n_channel)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_n_channel = out_n_channel

            x = x + identity
    return x

# Batch Normalization Layer
def batch_normalization(is_training, x, beta = None, gamma = None, epsilon = 1e-5, decay = 0.9,
                        name = 'BatchNormalization', op_collection = 'batch_normalization',
                        scope = None, reuse=False):
    """
    is_training: a placeholder indicating the training status
    x: input
    beta: numpy array
    gamma: numpy array
    """
    input_shape = x.get_shape().as_list()
    param_shape = input_shape[-1:]
    axis = list(range(len(input_shape) - 1))
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        
        if beta is None:
            beta = initialization('beta', shape = param_shape,
                                  initializer=tf.zeros_initializer())
        else:
            beta = initialization('beta', init_value=beta)

        if gamma is None: 
            gamma = initialization('gamma', shape = param_shape,
                                   initializer = tf.ones_initializer())   
        else:
            gamma = initialization('gamma', init_value=gamma)
        
        moving_mean = initialization('moving_mean', shape = param_shape, 
                                     initializer=tf.zeros_initializer(), trainable = False)
        moving_variance = initialization('moving_variance', shape = param_shape, 
                                         initializer=tf.ones_initializer(), trainable = False)
        
        def update():
            mean, variance = tf.nn.moments(x, axis)
    
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, 
                                                                       zero_debias = False)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay,
                                                                          zero_debias = False)
                
            tf.add_to_collection(op_collection, update_moving_mean)
            tf.add_to_collection(op_collection, update_moving_variance)
            
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(mean), tf.identity(variance)
            
        mean, variance = tf.cond(is_training, update, lambda:(moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)

        return x
    
# fully connected layer
def fullyconnected(x, 
                   W_init=None, W_shape=None, init_mean=0, init_std=0.02,
                   b_init=None, b_shape=None, 
                   regularizer=None, weight_decay=0.01,
                   name='FullyConnected', 
                   scope=None, reuse=False):
    """
    Arguments:
        x: input network;
        W_init/b_init: `numpy array`, initial value of W/b; 
        W_shape/b_shape: `list`, shape of W/b;
        W_initializer/b_initialier: `object`, tensorflow initializer;
        regularizer: `str`, 'L2';
        weight_decay" `float`;
    """    
    W_initializer = tf.random_normal_initializer(mean=init_mean, stddev=init_std)
    b_initializer = tf.random_uniform_initializer

    with tf.variable_scope(scope, default_name=name, reuse=reuse) as scope:
        W = initialization('W', 
                           init_value=W_init, shape=W_shape, initializer=W_initializer, 
                           regularizer=regularizer, weight_decay=weight_decay)
        b = initialization('b', 
                           init_value=b_init, shape=b_shape, initializer=b_initializer)
        
        x = tf.reshape(x, [-1, W.get_shape().as_list()[0]])
        x = tf.matmul(x, W)
        x = tf.add(x, b)
    return x

def dynamicRNN(x, seqlen, max_seq_len, n_hidden,
               unstack=True,
               W_init=None, W_shape=None, W_initializer=tf.random_normal_initializer,
               b_init=None, b_shape=None, b_initializer=tf.random_uniform_initializer,
               regularizer=None, weight_decay=0.01,
               trainable=True,
               name='LSTM', scope=None):
    """
    Argument:
        x: input data with shape (batch_size, n_steps, n_input)
           Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
       seqlen: `list`
       max_seq_len: `int`
       W_init/b_init:
       W_shape/b_shape:
       W_initializer/b_initializer:
       name: `str`
    """
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    if unstack:
        x = tf.unstack(x, max_seq_len, 1)
    
    # Create LSTM cell
    with tf.variable_scope(scope, default_name=name) as scope:        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)

        # change dimension of output of each time step to [batch_size, n_step, n_input]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        
        W = initialization('W', init_value=W_init, shape=W_shape, initializer=W_initializer, 
                           trainable=trainable, 
                           regularizer=regularizer, weight_decay=weight_decay)
        b = initialization('b', init_value=b_init, shape=b_shape, initializer=b_initializer, 
                           trainable=trainable)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        outputs = tf.matmul(outputs, W)
        outputs = tf.add(outputs, b, name='outputs')

    return outputs
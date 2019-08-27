import tensorflow as tf
from layers import *
    
def ConvNet(net, 
            n_layers = 1,
            params = None,
            filter_shape = None, strides = None, default_stride = 1, 
            initializer = None, trainable = True,
            regularizer=None, weight_decay=0.01,
            scope=None, name='Conv2D', reuse=False):
    
    if strides:
        if len(strides) != n_layers:
            myError = ValueError('len(strides) != n_layers')
        raise myError
        
    else:
        strides = [default_stride for i in range(n_layers)]
    if params:
        for i in range(n_layers):
            index = '_'.join([name, str(i)])
            W = params[index]['W']
            b = params[index]['b']
            if reuse:
                if i > 0:
                    scope = index
                else:
                    scope = name
            net = conv2d(net, 
                         W_init=W[0], b_init=b[0],
                         regularizer=regularizer, weight_decay=weight_decay,
                         strides=strides[i], 
                         scope=scope, name=name,
                         reuse=reuse)
        return net

    # Initializing weights and biases
    if not filter_shape:
        raise ValueError('The shape of filters must be defined') 

    for i in range(n_layers):
        W_shape = filter_shape[i]['W']
        b_shape = filter_shape[i]['b']
        if reuse:
            if i > 0:
                scope = index
            else:
                scope = name
        net = conv2d(net,
                     W_shape=W_shape, b_shape=b_shape,
                     regularizer=regularizer, weight_decay=weight_decay,
                     strides = strides[i], 
                     scope=scope, name=name,
                     reuse=reuse)
    return net

def ResNet(is_training, net, n_blocks,
           params=None, downsample=None, 
           filter_shape=None, strides=None, downsample_stride=None, default_stride=1,
           regularizer=None, weight_decay=0.01,
           beta=None, gamma=None,
           padding=None, scope=None,
           name='ResidualBlock',
           reuse=False, bias=True):
    
    n_layers = len(n_blocks)
    if downsample:
        if len(downsample) != n_layers:
            myError = ValueError('len(downsample) != n_layers')
            raise myError
    else:
        downsample = [False for i in range(n_layers)]

    default_strides = [default_stride, default_stride]
    downsample_strides = [downsample_stride, default_stride]
    strides = [downsample_strides if downsample[i] else default_strides for i in range(n_layers)]
    
    if params:
        for i in range(n_layers):
            index = '_'.join([name, str(i)])
        
            W_init = params[index]['W']
            b_init = params[index].get('b', None)
            beta = params[index]['beta']
            gamma = params[index]['gamma']
            if reuse:
                if i > 0:
                    scope = index
                else:
                    scope = name
            net = residual_block(is_training, net, 
                                 n_blocks=n_blocks[i], all_strides=strides[i],
                                 W_init=W_init, b_init=b_init,
                                 beta=beta, gamma=gamma,
                                 regularizer=regularizer, weight_decay=weight_decay,
                                 scope=scope, name=name,
                                 reuse=reuse, bias=bias)
        return net
    
    # Initializing weights and biases
    if not filter_shape:
        raise ValueError('The shape of filters must be defined') 

    for i in range(n_layers):
        index = '_'.join([name, str(i)])
        W_shape_0 = filter_shape[i]['W0']
        b_shape_0 = filter_shape[i].get('b0', None)

        W_shape_1 = filter_shape[i].get('W1', None)
        b_shape_1 = filter_shape[i].get('b1', None)

        if reuse:
            if i > 0:
                scope = index
            else:
                scope = name
                
        net = residual_block(is_training, net,
                             n_blocks=n_blocks[i], all_strides=strides[i],
                             W_shape_0=W_shape_0, W_shape_1=W_shape_1,
                             b_shape_0=b_shape_0, b_shape_1=b_shape_1,
                             regularizer=regularizer, weight_decay=weight_decay, 
                             scope=scope, name=name,
                             reuse=reuse, bias=bias)
    return net
        
def FullyConnected(net, params, 
                   n_layers=1,
                   regularizer=None, weight_decay=0.01,
                   default_name='ResidualBlock',
                   scope=None, name=None,
                   reuse=False):
    
    for i in range(n_layers):
        index = '_'.join(['FullyConnected', str(i)])
        
        W = params[index]['W'][0]
        b = params[index]['b'][0]
        
        if reuse:
            if i > 0:
                scope = index
            else:
                scope = name
        net = fullyconnected(net, W, b, scope=scope, name=name, reuse=reuse)
      
    return net
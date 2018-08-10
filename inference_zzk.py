import tensorflow as tf
import cv2
import numpy as np


IMG_SIZE = 64
NUM_CHANNELS = 3
# 第一层卷积层的尺寸与深度
CONV1_SIZE = 5
CONV1_DEPTH = 18
# 第二层
CONV2_SIZE = 5
CONV2_DEPTH = 36
# 第三层
CONV3_SIZE = 3
CONV3_DEPTH = 64
# 第四层
CONV4_SIZE = 3
CONV4_DEPTH = 108
# 全连接层1
FC1_SIZE = 256
FC2_SIZE = 64
OUTPUT_SIZE = 2


def get_weight(shape):
    return tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def get_bias(shape):
    return tf.get_variable("bias", shape, initializer=tf.constant_initializer(0.0))


def inference(input_tensor, train, regularizer):
    # 第一层，输出32*32*conv1_depth
    with tf.variable_scope("conv1"):
        conv1_weights = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEPTH])
        conv1_biases = get_bias([CONV1_DEPTH])
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1],padding='SAME')
        # same padding in tensorflow意味着使输出维度为int(img_size/stride)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope("pooling1"):
        pooling1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], 
                        strides=[1, 2, 2, 1], padding='SAME')
    # 第二层输出16*16*conv2_depth
    with tf.variable_scope("conv2"):
        conv2_weights = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH])
        conv2_biases = get_bias([CONV2_DEPTH])
        conv2 = tf.nn.conv2d(pooling1, conv2_weights, strides=[1,1,1,1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope("pooling2"):
        pooling2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # 第三层，输出8*8*conv3_depth
    with tf.variable_scope("conv3"):
        conv3_weights = get_weight([CONV3_SIZE, CONV3_SIZE, CONV2_DEPTH, CONV3_DEPTH])
        conv3_biases =get_bias([CONV3_DEPTH])
        conv3 = tf.nn.conv2d(pooling2, conv3_weights, strides=[1,1,1,1], padding='SAME')
        relu3 = tf.nn/relu(tf.nn.bias_add(conv3, conv3_biases))
    with tf.name_scopt("pooling3"):
        pooling3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # 第四层 4*4*conv4_depth(4*4*108)
    with tf.variable_scope("conv4"):
        conv4_weights = get_weight([CONV4_SIZE, CONV4_SIZE, CONV3_DEPTH, CONV4_DEPTH])
        conv4_biases =get_bias([CONV4_DEPTH])
        conv4 = tf.nn.conv2d(pooling3, conv4_weights, strides=[1,1,1,1], padding='SAME')
        relu4 = tf.nn/relu(tf.nn.bias_add(conv4, conv4_biases))
    with tf.name_scopt("pooling4"):
        pooling4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


    conv_out_shape = pooling4.get_shape().as_list()
    nodes1 = conv_out_shape[1]*conv_out_shape[2]*conv_out_shape[3]
    fc_input = tf.reshape(pooling4, [conv_out_shape[0], nodes])
    # 第一层全连接
    with tf.variable_scope('fc_layer_1'):
        fc_weights_1 = get_weight([nodes, FC1_SIZE])
        if regularizer !=None:
            tf.add_to_collection("loss", regularizer(fc_weights_1))
        fc_biases_1 = get_bias([FC1_SIZE])
        fc_layer_1 = tf.nn.relu(tf.matmul(fc_input, fc_weights_1)+fc_biases_1)
        # 只在全连接层且是训练时才需要dropout
        if train: fc_layer_1 = tf.nn.dropout(fc_layer_1, 0.5)
    # 第二层全连接
    with tf.variable_op_scope("fc_layer_2"):
        fc_weights_2 = get_weight([FC1_SIZE, FC2_SIZE])
        if regularizer !=None:
            tf.add_to_collection("loss", regularizer(fc_weights_2))
        fc_biases_2 = get_bias([FC2_SIZE])
        fc_layer_2 = tf.nn.relu(tf.matmul(fc_layer_1, fc_weights_2)+fc_biases_2)
        # 只在全连接层且是训练时才需要dropout
        if train: fc_layer_2 = tf.nn.dropout(fc_layer_2, 0.5)
    # 第三次全连接
    with tf.variable_op_scope("fc_layer_3"):
        fc_weights_3 = get_weight([FC2_SIZE, OUTPUT_SIZE])
        if regularizer !=None:
            tf.add_to_collection("loss", regularizer(fc_weights_3))
        fc_biases_3 = get_bias([OUTPUT_SIZE])
        logit = tf.matmul(fc_layer_2, fc_weights_3) + fc_biases_3
    return logit
    
    
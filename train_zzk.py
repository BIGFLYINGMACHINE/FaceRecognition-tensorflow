import tensorflow as tf
import input_data_zzk as input_data
import inference_zzk as infer


BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model_zzk"
MODE_NAME = "model.ckpt"



def train(faces):
    x = tf.placeholder(tf.float32, [None, size, size, 3], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    relularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = infer.inference(x, train=False) # 未训练时不设为True
    global_step =tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, 
                                                        labels=tf.arg_max(y_, 1))
    
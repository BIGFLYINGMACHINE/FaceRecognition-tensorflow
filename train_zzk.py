import tensorflow as tf
import numpy as np
import input_data_zzk as input_data
import inference_zzk as infer
import os

BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model_zzk"
MODE_NAME = "model.ckpt"
DECAY_STEPS = 400 # 与学习率的指数衰减有关，此数值越大衰减越慢
train_x, test_x, train_y, test_y = input_data.get_data()


def next_batch(num, data, labels):
    '''
    返回batch_size 为 num 的batch
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train():
    x = tf.placeholder(tf.float32, [None, size, size, 3], name='x-input') 
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    relularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    y = infer.inference(x, train=False) # 未训练时不设为True
    global_step =tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, 
                                                        labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数
    losses = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
                                                DECAY_STEPS, LEARNING_RATE_DECAY)
    train_step =tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
    # 下面两行不知何用
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = next_batch(BATCH_SIZE, train_x, train_y)
            _, loss_value, step = sess.run([train_op, losses, global_step],
                                    feed_dict={x:xs, y_:ys})
            if i%500 == 0:
                print("After %d training steps, loss on training batch in %g"%(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                                                global_step=global_step)


def main():
    train()


if __name__ == "__main__":
    tf.app.run()

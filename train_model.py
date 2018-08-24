import tensorflow as tf
import numpy as np
import input_data as input_data
import inference as infer
import os
import sys
from parameters import *
train_x, test_x, train_y, test_y = input_data.get_data()


def next_batch(num, data, labels):
    '''
    返回batch_size 为 num 的batch
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train():
    x = tf.placeholder(
        tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
    # 训练时, train=True，则会启动dropout
    # 给regularizer赋值则会将加入正则化，weights被正则化后加入集合losses当中
    y = infer.inference(x, train=True, regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 准确率计算
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
                                                                   labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 损失函数
    losses = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               DECAY_STEPS, LEARNING_RATE_DECAY)
    train_step = tf.train.AdamOptimizer(
        learning_rate).minimize(losses, global_step=global_step)
    # 下面两行不知何用
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        outcome = {}  # 初始化输出结果的字典，之后会被保存为csv文件
        for i in range(TRAINING_STEPS):
            xs, ys = next_batch(BATCH_SIZE, train_x, train_y)
            _, loss_value, step = sess.run([train_op, losses, global_step],
                                           feed_dict={x: xs, y_: ys})
            if i % 300 == 0:
                accuracy_train = sess.run(accuracy,
                                          feed_dict={x: xs, y_: ys})
                accuracy_test = sess.run(accuracy,
                                         feed_dict={x: test_x, y_: test_y})
                outcome['train'+str(step)] = accuracy_train
                outcome['test'+str(step)] = accuracy_test
                save_parameters_as_csv(outcome, "./paras.csv")
                print("After %d steps, accuracy on train set is %g" %
                      (step, accuracy_train))
                print("After %d steps, accuracy on test set is %g" %
                      (step, accuracy_test))
                print("After %d steps, loss on training batch is %g" %
                      (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)
        if KeyboardInterrupt:
            sys.exit()

def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
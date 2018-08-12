import tensorflow as tf
import input_data_zzk as input_data

import inference_zzk as infer
import train_zzk as train
from parameters import *

train_x, test_x, train_y, test_y = input_data.get_data()


def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(
            tf.float32, [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS],
                                                name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

        validate_feed = {x: test_x, y_: test_y}
        y = infer.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./model_zzk"))
            accuracy_score = sess.run(accuracy, 
                                        feed_dict=validate_feed)
            print("After training steps, validation "
                    "accuracy = %g"%(accuracy_score))


def main(argv=None):
    evaluate()


if __name__ == "__main__":
    tf.app.run()
    
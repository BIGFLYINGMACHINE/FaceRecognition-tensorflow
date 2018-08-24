import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import inference as infer
import input_data
IMAGE_SIZE = 64
path = "D:/Total/own_face_recognition/dataset/faces/test_mine"



def test_on_camera():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], name="x")
        out = infer.inference(x, train=False, regularizer=None)
        predict = tf.argmax(out, 1)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint("./model")) 
            #使用dlib自带的frontal_face_detector作为我们的特征提取器
            detector = dlib.get_frontal_face_detector()

            cam = cv2.VideoCapture(0)  
            while True:  
                _, img = cam.read()  
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_image, 1)
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1,x2:y2]
                    # 调整图片的尺寸
                    face = cv2.resize(face, (IMAGE_SIZE,IMAGE_SIZE))
                    input_image = np.array(face)/255.0
                    input_image.astype('float32')
                    res = sess.run(predict, feed_dict={x: [input_image]})
                    sess.run(out, feed_dict={x: [input_image]})  
                    print(res)
                    print(out)
                    if res[0] == 1:  
                        # 是本人
                        cv2.rectangle(img, (x2,x1),(y2,y1), (0,0,255),3)

                    else:  
                        # 不是本人
                        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
                    
                cv2.imshow('image',img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)

def test_on_picture():
    train_x, test_x, train_y, test_y = input_data.get_data()
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], name="x")
        out = infer.inference(x, train=False, regularizer=None)
        predict = tf.argmax(out, 1)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint("./model")) 
            res = sess.run(predict, feed_dict={x: test_x})
            print(res)


test_on_camera()
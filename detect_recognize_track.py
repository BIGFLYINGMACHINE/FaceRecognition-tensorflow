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

with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 64, 64, 3], name="x")
        out = infer.inference(x, train=False, regularizer=None)
        predict = tf.argmax(out, 1)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint("./model")) 
            #使用dlib自带的frontal_face_detector作为我们的特征提取器
            detector = dlib.get_frontal_face_detector()
            face_comfirmed = False
            cam = cv2.VideoCapture(0)  
            while not face_comfirmed:  
                _, img = cam.read()  
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_image, 1)
                for i, d in enumerate(dets):
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1,x2:y2]
                    bbox = (x1, y1, x2, y2)
                    # 调整图片的尺寸
                    input_image = cv2.resize(face, (IMAGE_SIZE,IMAGE_SIZE))
                    input_image = np.array(input_image)/255.0
                    input_image.astype('float32')
                    res = sess.run(predict, feed_dict={x: [input_image]})
                    sess.run(out, feed_dict={x: [input_image]})  
                    print(res)
                    print(out)
                    if res[0] == 1:  
                        # 是本人
                        cv2.rectangle(img, (x2,x1),(y2,y1), (0,0,255),3)

                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(img, bbox)
                        face_comfirmed = True
                cv2.imshow('image',img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)
            while True:
                timer = cv2.getTickCount()
                # 更新
                ok, bbox = tracker.update(img)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
                # Display tracker type on img
                cv2.putText(img,  "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
            
                # Display FPS on img
                cv2.putText(img, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
                # Display result
                cv2.imshow("Tracking", img)
        
                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
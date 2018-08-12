import os 
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = "D:/Total/own_face_recognition/dataset/"
classes = {'my_faces', 'other_faces'} #预先自己定义的类别
writer = tf.python_io.TFRecordWriter('train.tfrecords') #输出成tfrecord文件

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

for index, name in enumerate(classes):
    class_path = cwd + name + '//'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name    #每个图片的地址

        img = Image.open(img_path)
        img = img.resize((208, 208))
        img_raw = img.tobytes()  #将图片转化为二进制格式
        example = tf.train.Example(features = tf.train.Features(feature = {
                                                                           "label": _int64_feature(index),
                                                                           "img_raw": _bytes_feature(img_raw),                                                                          
                                                                           }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()
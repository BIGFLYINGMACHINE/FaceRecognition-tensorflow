import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = 'D:/Total/own_face_recognition/dataset/faces/my_faces'
other_faces_path = 'D:/Total/own_face_recognition/dataset/faces/other_faces'
size = 64


def readData(path, h=size, w=size):
    global imgs
    global labs
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            imgs.append(img)
            labs.append(path)

def get_data():
    labs = []
    imgs = []
    for filename in os.listdir(my_faces_path):
        if filename.endswith('.jpg'):
            filename = my_faces_path + '/' + filename
            img = cv2.imread(filename)
            imgs.append(img)
            # 如果是“my_face”便label为[0, 1],否则label为[1, 0]
            labs.append([0, 1])
    for filename in os.listdir(other_faces_path):
        if filename.endswith('.jpg'):
            filename = other_faces_path + '/' + filename
            img = cv2.imread(filename)
            imgs.append(img)
            labs.append([1, 0])
    # 将图片数据与标签转换成数组
    imgs = np.array(imgs)

    
    labs = np.array(labs)

    # 随机划分测试集与训练集
    train_x, test_x, train_y, test_y = train_test_split(
        imgs, labs, test_size=0.2, random_state=random.randint(0, 100))
    # 参数：图片数据的总数，图片的高、宽、通道
    # ?疑问，下面两行真的改变了shape吗?
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # 将数据转换成小于1的数, feature scaling
    train_x = train_x.astype('float32')/255.0
    test_x = test_x.astype('float32')/255.0
    return train_x, test_x, train_y, test_y

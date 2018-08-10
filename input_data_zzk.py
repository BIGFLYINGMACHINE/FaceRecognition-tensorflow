import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = './my_faces'
other_faces_path = './other_faces'
size = 64

imgs = []
labs = []


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            #
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)

def get_data():
    readData(my_faces_path)
    readData(other_faces_path)
    # 将图片数据与标签转换成数组
    imgs = np.array(imgs)

    # 如果是“my_face”便label为[0, 1],否则label为[1, 0]
    labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

    # 随机划分测试集与训练集
    train_x, test_x, train_y, test_y = train_test_split(
        imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
    # 参数：图片数据的总数，图片的高、宽、通道
    # ?疑问，下面两行真的改变了shape吗?
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)
    # 将数据转换成小于1的数, feature scaling
    train_x = train_x.astype('float32')/255.0
    test_x = test_x.astype('float32')/255.0
    return train_x, test_x, train_y, test_y

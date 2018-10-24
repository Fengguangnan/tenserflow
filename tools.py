import struct as st
import numpy as np
import cv2
import os
import shutil


def create_sub_dir(path):
    count = 0
    while count < 10:
        dir_name = os.path.join(path, str(count))
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)
        count += 1
    return

def get_data(filename):

    fp = open(filename, 'rb')
    data = fp.read()
    fp.close()

    head = st.unpack_from('>4I', data, 0)   # 获得头部16字节的信息
    offset = st.calcsize('>4I')             # 头部16字节后是图片数据
    images = head[1]                        # 图片数量
    w = head[2]                             # 图片宽度
    h = head[3]                             # 图片高度

    format_str = '>' + str(images * w * h) + 'B'
    data = st.unpack_from(format_str, data, offset)
    return data


def get_label(filename):

    fp = open(filename, 'rb')
    data = fp.read()
    fp.close()

    head = st.unpack_from('>2I', data, 0)   # 从偏移地址0处以大端模式读取两个u32类型的数据
    label_num = head[1]                     # 获得标签数量
    offset = st.calcsize('>2I')             # 标签数据的偏移

    format_str = '>' + str(label_num) + 'B'
    labels = st.unpack_from(format_str, data, offset)
    return labels


def load_mnist(path):

    data_train = get_data(os.path.join(path, 'train-images.idx3-ubyte'))
    label_train = get_label(os.path.join(path, 'train-labels.idx1-ubyte'))

    data_eval = get_data(os.path.join(path, 't10k-images.idx3-ubyte'))
    label_eval = get_label(os.path.join(path, 't10k-labels.idx1-ubyte'))

    data_train = np.array(data_train).reshape(60000, 28, 28)
    data_eval = np.array(data_eval).reshape(10000, 28, 28)

    label_train = np.array(label_train).reshape(60000)
    label_eval = np.array(label_eval).reshape(10000)

    return (data_train, label_train), (data_eval, label_eval)

def load_mnist_eval(path):
    data = get_data(os.path.join(path, 't10k-images.idx3-ubyte'))
    label = get_label(os.path.join(path, 't10k-labels.idx1-ubyte'))
    data = np.array(data).reshape(10000, 28, 28)
    label = np.array(label).reshape(10000)
    return (data, label)


def load_user_images(path):
    images = []
    labels = []
    images_name = os.listdir(path)

    for fname in images_name:
        pathname = os.path.join(path, fname)
        img = cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)        # 以灰度方式读取图片
        cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)         # 二值化图片
        cv2.bitwise_not(img, img)                               # 颜色反转
        img = cv2.resize(src=img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)  # 改变图片大小到28 * 28
        images.append(img)
        labels.append(int(fname[:-4]))

    labels = np.array(labels).reshape(len(labels))
    images = np.array(images).reshape(len(images), 28, 28)

    return (images, labels)

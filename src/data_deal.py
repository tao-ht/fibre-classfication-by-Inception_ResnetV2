import os
import numpy as np
import random
import cv2

# 训练datasets图片的存放路径
TRAIN_IMAGE_PATH = "../datasets/train_299_299/"
# 图片名存放路径
train_file_dir = '../datasets/train_name_label.csv'
val_file_dir = "../datasets/val_name_label.csv"

IMAGE_WIDHT = 299
IMAGE_HEIGHT = 299

"""resize到统一尺寸200*200"""
def resize_norl(Image_array):
    data = []
    for img in Image_array[:]:
        weight, height = img.shape[0:2]
        ratio = weight/height
        if ratio == 1:
            img = cv2.resize(img, (200, 200))
        elif ratio > 1:
            r = weight/200
            h = int(height // r)
            img = cv2.resize(img, (200, h))
            h_l = int((200-h)//2)
            h_r = 200-h-h_l
            img = np.pad(img, ((h_l, h_r), (0, 0), (0, 0)), 'constant')
        elif ratio < 1:
            r = height/200
            w = int(weight//r)
            img = cv2.resize(img, (w, 200))
            w_l = int((200-w)//2)
            w_r = 200-w-w_l
            img = np.pad(img, ((0, 0), (w_l, w_r), (0, 0)), 'constant')
        data.append(img)
    data = np.array(data)
    return data

"""load train or val pictures data,get name and labels"""
def pic_name_label_load(train_file, val_file):
    val_name, train_name = [], []
    val_label, train_label = [], []
    with open(train_file, "r") as file_in:
        pics_name = file_in.readlines()
        random.shuffle(pics_name)
        for i in range(len(pics_name)):
                train_name.append(pics_name[i].split(" ", 1)[0])
                train_label.append(int(pics_name[i].split(" ", 1)[1]))  # 切片提取label，剔除换行符)[:-1]
    with open(val_file, "r") as file_in:
        pics_name = file_in.readlines()
        random.shuffle(pics_name)
        for i in range(len(pics_name)):
            val_name.append(pics_name[i].split(" ", 1)[0])
            val_label.append(int(pics_name[i].split(" ", 1)[1]))  # 切片提取label，剔除换行符)[:-1]
    train_name = np.array(train_name).reshape(-1, 1)
    train_label = np.array(train_label).reshape(-1, 1)
    val_name = np.array(val_name).reshape(-1, 1)
    val_label = np.array(val_label).reshape(-1, 1)
    return train_name, train_label, val_name, val_label

def onehot_encoding(Y_data, classes=3):
    Y_label = np.zeros((Y_data.shape[0], classes), dtype=int)
    for i in range(Y_data.shape[0]):
        Y_label[i, Y_data[i, ]] = 1
    return Y_label

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def color_preprocessing(x_train):#, x_test
    x_train = x_train.astype('float32')

    # # x = x-[x（均值）/x（标准差）]
    # x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    # x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    # x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
    return x_train#/255.0

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    # batch = _random_crop(batch, [299, 299], 4)
    return batch

def prepare_data():
    train_name, train_label, val_name, val_label = pic_name_label_load(train_file_dir, val_file_dir)
    return train_name, train_label, val_name, val_label


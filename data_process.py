import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import cv2

parent_path = './ADNI'
parent_path = os.path.abspath(parent_path)

user_ids = os.listdir(parent_path)

resize_shape = (192, 192)

user_data = {}

scaler = MinMaxScaler(feature_range=(0,1))

def get_nii_path(user_id):
    path = os.path.join(parent_path, user_id)
    for _ in range(4):
        path = os.path.join(path, os.listdir(path)[0])
    return path

def resize_img(img):
    return cv2.resize(img, resize_shape)

def gif_show(img, id):
    num_images = list(img.shape)[-1]
    for i in range(num_images):
        # img_n = scaler.fit(img[:, :, i])
        # print("New shape: ",img_n.shape)
        plt.imshow(img[:, :, i], cmap='gray')
        plt.show()

for id in user_ids[4:]:
    nii_path = get_nii_path(id)
    nii_file = nib.load(nii_path)
    data = nii_file.get_data()
    data = np.asarray(data)
    data = np.squeeze(data)
    data = resize_img(data).astype(np.float32)
    user_data[id] = data
    print("{}, Shape: {}".format(id, data.shape))
    # print(np.max(data))
    # print(np.min(data))
    gif_show(data, id)
    break

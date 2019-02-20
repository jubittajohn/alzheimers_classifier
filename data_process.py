import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from xml.dom import minidom

import cv2

parent_path = './ADNI'
parent_path = os.path.abspath(parent_path)

user_ids = os.listdir(parent_path)

resize_shape = (228,228)

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
small_3d=10000
for id in user_ids:
    nii_path = get_nii_path(id)
    nii_file = nib.load(nii_path)
    data = nii_file.get_data()
    data = np.asarray(data)
    data = np.squeeze(data)
   # data = resize_img(data).astype(np.float32)
    user_data[id] = data
    print("{}, Shape: {}".format(id, data.shape))
    if data.shape[2]<small_3d:
            small_3d=data.shape[2]
    # print(np.max(data))
    # print(np.min(data))
    #gif_show(data, id)
   # break
#making thrid dimension same by slicing from begining and end
print("------------------")
for id in user_ids:
        data=user_data[id]
        diff_3d = data.shape[2] - small_3d
        begining_diff = int(diff_3d/2)
        end_diff = diff_3d - begining_diff
        if diff_3d != 0:
                data=data[:,:,begining_diff.__index__():]
                data=data[:,:,:-end_diff.__index__()]
        user_data[id]=data
        print("{}, Shape: {}".format(id, data.shape))
gif_show(data, id)

#get reasearch group from metadata
parent_path = './ADNI_metadata'
parent_path = os.path.abspath(parent_path)

user_research_grp={}
user_idnames = os.listdir(parent_path)
for user_id in user_idnames:
        path=os.path.join(parent_path,user_id)
        xmldoc = minidom.parse(path)
        id=xmldoc.getElementsByTagName("subjectIdentifier")[0].firstChild.data
        reasearch_grp=xmldoc.getElementsByTagName("researchGroup")[0].firstChild.data
        user_research_grp[id]=reasearch_grp

print("------------------")
for id in user_ids:
         print("{}, Shape: {},Research Group: {}".format(id, user_data[id].shape,user_research_grp[id]))



import os
import sys

#import matplotlib.animation as animation
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.animation import PillowWriter
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

from xml.dom import minidom

import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, MaxPooling3D, ZeroPadding3D
from keras import losses

parent_path = './ADNI'
parent_path = os.path.abspath(parent_path)

user_ids = os.listdir(parent_path)

user_data = {}

scaler = MinMaxScaler(feature_range=(0,1))

#Obtaining path to the dataset
def get_nii_path(user_id):
    path = os.path.join(parent_path, user_id)
    for _ in range(4):
        path = os.path.join(path, os.listdir(path)[0])
    return path


#Display MRI
def gif_show(img, id):
    num_images = list(img.shape)[-1]
    for i in range(num_images):
        plt.imshow(img[:, :, i], cmap='gray')
        plt.show()
small_3d=10000

#Storing as a Numpy array
for id in user_ids:
    nii_path = get_nii_path(id)
    nii_file = nib.load(nii_path)
    data = nii_file.get_data()
    data = np.asarray(data)
    data = np.squeeze(data)
    user_data[id] = data
#     print("{}, Shape: {}".format(id, data.shape))
    if data.shape[2]<small_3d:
            small_3d=data.shape[2]

#Making thrid dimension same by slicing from begining and end
print("------------------")
x_input = []
for id in user_ids:
        data=user_data[id]
        diff_3d = data.shape[2] - small_3d
        begining_diff = int(diff_3d/2)
        end_diff = diff_3d - begining_diff
        if diff_3d != 0:
                data=data[:,:,begining_diff.__index__():]
                data=data[:,:,:-end_diff.__index__()]
        user_data[id]=data
        x_input.append(data)
        # print("{}, Shape: {}".format(id, data.shape))

#Get reasearch group from metadata
parent_path = './ADNI_metadata'
parent_path = os.path.abspath(parent_path)

user_research_grp={}
user_idnames = os.listdir(parent_path)
for user_id in user_idnames:
        path = os.path.join(parent_path,user_id)
        xmldoc = minidom.parse(path)
        id = xmldoc.getElementsByTagName("subjectIdentifier")[0].firstChild.data
        reasearch_grp = xmldoc.getElementsByTagName("researchGroup")[0].firstChild.data
        user_research_grp[id] = reasearch_grp

print("------------------")
# for id in user_ids:
#          print("{}, Shape: {},Research Group: {}".format(id, user_data[id].shape,user_research_grp[id]))

y_input = []
for id in user_ids:
        print(id)
        y_input.append(user_research_grp[id])


#One hot encoding of the data
integer_encoded = LabelEncoder().fit_transform(y_input)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("Shape of One Hot Encoding: {}".format(np.shape(onehot_encoded)))

x_input = np.expand_dims(x_input, axis=4)
print("Shape of input data: {}".format(np.shape(x_input)))

#A simple NN architecture
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(256, 256, 44, 1)))
model.add(Activation('relu'))
model.add(Conv3D(32, kernel_size=(3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(3, 3, 3)))
model.add(Activation('relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss=losses.binary_crossentropy, optimizer="adam", metrics=['accuracy'])
history = model.fit(x_input, onehot_encoded, batch_size=2, epochs=15)
print(history)

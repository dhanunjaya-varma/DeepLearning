#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:11:40 2019

@author: dhanunjaya
"""

import numpy as np
import cv2
from keras.models import load_model
import os
from PIL import Image

#img_test=np.load('./img_test.npy')
#mask_test=np.load('./mask_test.npy')

#mask_test = np.reshape(mask_test, (len(mask_test), 256, 256, 1))

#mask_test = mask_test.astype('float32') / 255
#img_test = img_test.astype('float32') / 255

print("enter the folder path for the images whose mask has to be predicted:")
input_path=input()
print(input_path)

data_files = os.listdir(input_path)

for i in range(len(data_train)):
    img_path=data_dir+'/'+data_train[i]
    img=cv2.imread(img_path)
    orgX=img.shape[0]
    orgY=img.shape[1]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    thresh = np.expand_dims(thresh, axis=2)
    img=cv2.resize(thresh,(256,256))
    img_train.append(img)

img_train = np.expand_dims(img_train, axis = 3)
img_train = img_train.astype('float32') / 255

if os.path.exists("./path_to_testing_folder"):
    print("folder exists")
else:
    print("folder doesn't exist")
    os.mkdir("./path_to_testing_folder")
    output_path=os.path.abspath("./path_to_testing_folder")
    print(output_path)

img_train = []

model=load_model('./core_point.h5')
pred = model.predict(img_train)

for i in range(len(data_train)):
    pred = pred[i]*256
    pred=np.squeeze(pred, axis=2)
    mask=cv2.resize(pred,(orgY,orgX))
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    M = cv2.moments(pred)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    lables=data_train[i].split('jpeg')
    

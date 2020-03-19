#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:48:15 2019

@author: dhanunjaya
"""

import os
import cv2
import numpy as np
from keras import models
from PIL import Image

print("enter the folder path for the images whose mask has to be predicted:")
input_path=input()
print(input_path)

if os.path.exists("./predicted_mask"):
    print("folder exists")
else:
    print("folder doesn't exist")
    os.mkdir("./predicted_mask")
    output_path=os.path.abspath("./predicted_mask")
    print(output_path)

eye_files = []
eye_train = []
eye_files = os.listdir(input_path)

for i in range(len(eye_files)):
    eye_name=eye_files[i]
    eye_path=input_path+'/'+str(eye_name)
    eye_img=cv2.imread(eye_path)
    eye_img=cv2.resize(eye_img,(128,128))
    eye_train.append(eye_img)

eye_train = np.asarray(eye_train, dtype=np.float32)
eye_train = eye_train.astype('float32') / 255

model= models.load_model('/home/dhanunjaya/Downloads/DL/Assignment_3/Assignment3/Q2/resize_auto_imgSeg.h5')

predictions = model.predict(eye_train)

for i in range(eye_train.shape[0]):
    p2=np.reshape(predictions[i], (128,128))
    p2 = p2 *255
    p2 = p2.astype('int32')
    img=Image.fromarray(p2)
    lables=eye_files[i].split('original')
    mask_name='Predicted_'+lables[1]
    cv2.imwrite(output_path+'/'+mask_name, p2)
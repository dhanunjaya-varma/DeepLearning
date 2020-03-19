#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 23:49:38 2019

@author: dhanunjaya
"""

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

eye_train = []
mask_train = []
path_eye = './Q2/Data'
path_mask = './Q2/Mask'
eye_files = os.listdir(path_eye)
mask_files = os.listdir(path_mask)
eye_files = sorted(eye_files)
mask_files = sorted(mask_files)

eye_train_img, eye_test_img, mask_train_img, mask_test_img = train_test_split(eye_files, mask_files, test_size=0.3)

eye_train = []
eye_test = []
mask_train = []
mask_test = []

for i in range(len(eye_train_img)):
    eye_name=eye_train_img[i]
    eye_path=path_eye+'/'+str(eye_name)
    eye_img=cv2.imread(eye_path)
    eye_img=cv2.resize(eye_img,(128,128))
#    eye_img=cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
#    eye_img = eye_img.astype('float32') / 255
    eye_train.append(eye_img)
#    print(eye_name)
    
    mask_name=mask_train_img[i]
    mask_path=path_mask+'/'+str(mask_name)
    mask_img=cv2.imread(mask_path)
    mask_img=cv2.resize(mask_img,(128,128))
    mask_img=cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
#    mask_img = mask_img.astype('float32') / 255
    mask_train.append(mask_img)
#    print(mask_name)

for i in range(len(eye_test_img)):
    eye_name=eye_test_img[i]
    eye_path=path_eye+'/'+str(eye_name)
    eye_img=cv2.imread(eye_path)
    eye_img=cv2.resize(eye_img,(128,128))
#    eye_img=cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
#    eye_img = eye_img.astype('float32') / 255
    eye_test.append(eye_img)
#    print(eye_name)
    
    mask_name=mask_test_img[i]
    mask_path=path_mask+'/'+str(mask_name)
    mask_img=cv2.imread(mask_path)
    mask_img=cv2.resize(mask_img,(128,128))
    mask_img=cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
#    mask_img = mask_img.astype('float32') / 255
    mask_test.append(mask_img)
#    print(mask_name)
 
np.save('./Q2/eye_train_sz.npy', eye_train) 
np.save('./Q2/mask_train_sz.npy', mask_train)
np.save('./Q2/eye_test_sz.npy', eye_test)
np.save('./Q2/mask_test_sz.npy', mask_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:09:14 2019

@author: dhanunjaya
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

root_dir='./Q3/Core_Point'
data_dir=root_dir+'/Data'
gt_dir=root_dir+'/Ground_truth'
img_train = []
img_test = []
mask_train = []
mask_test = []

data_files = os.listdir(data_dir)
gt_files = os.listdir(gt_dir)
data_files = sorted(data_files)
gt_files = sorted(gt_files)

data_train, data_test, gt_train, gt_test = train_test_split(data_files, gt_files, test_size=0.25)

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
    
    mask_path=gt_dir+'/'+gt_train[i]
    with open(mask_path) as fp:
        lines1 = fp.readlines()
    lines_train = [x.strip() for x in lines1]
    lines_split=lines_train[0].split(' ')
    cy=int(lines_split[0])
    cx=int(lines_split[1])
    mask = np.zeros([orgX, orgY], dtype = 'uint8')
    mask=cv2.circle(mask, (cx, cy), 30, (255, 255, 255), -1)
    mask=cv2.resize(mask,(256,256))
    mask_train.append(mask)
    
    
for i in range(len(data_test)):
    img_path=data_dir+'/'+data_test[i]
    img=cv2.imread(img_path)
    orgX=img.shape[0]
    orgY=img.shape[1]
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    thresh = np.expand_dims(thresh, axis=2)
    img=cv2.resize(thresh,(256,256))
    img_test.append(img)
    
    mask_path=gt_dir+'/'+gt_test[i]
    with open(mask_path) as fp:
        lines1 = fp.readlines()
    lines_test = [x.strip() for x in lines1]
    lines_split=lines_test[0].split(' ')
    cy=int(lines_split[0])
    cx=int(lines_split[1])
    mask = np.zeros([orgX, orgY], dtype = 'uint8')
    mask=cv2.circle(mask, (cx, cy), 30, (255, 255, 255), -1)
    mask=cv2.resize(mask,(256,256))
    mask_test.append(mask)
    
np.save('./img_train.npy',img_train)
np.save('./img_test.npy',img_test)
np.save('./mask_train.npy',mask_train)
np.save('./mask_test.npy',mask_test)

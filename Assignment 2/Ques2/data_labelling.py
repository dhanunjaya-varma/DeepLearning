#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 00:38:48 2019

@author: dhanunjaya
"""

import cv2
import os
import numpy as np

x_train = []
length_train = []
width_train = []
angles_train = []
color_train = []

x_test = []
length_test = []
width_test = []
angles_test = []
color_test = []

root_dir = os.getcwd()
path_train=os.path.join(root_dir,'train/')
path_test=os.path.join(root_dir,'test/')
train_files = os.listdir(path_train)
test_files = os.listdir(path_test)

for i in range(len(train_files)):
    img_dir=os.path.join(path_train,train_files[i])
    images=os.listdir(img_dir)
    for j in range(len(images)):
        img_name=images[j]
        img_path=img_dir+'/'+str(img_name)
        lables=img_name.split('_')
        img=cv2.imread(img_path)
        x_train.append(img)
        length_train.append(int(lables[0],10))
        width_train.append(int(lables[1],10))
        angles_one_hot_train = np.eye(12)[int(lables[2],10)]
        angles_train.append(angles_one_hot_train)
        color_train.append(int(lables[3],10))


for i in range(len(test_files)):
    img_dir=os.path.join(path_test,test_files[i])
    images=os.listdir(img_dir)
    for j in range(len(images)):
        img_name=images[j]
        img_path=img_dir+'/'+str(img_name)
        lables=img_name.split('_')
        img=cv2.imread(img_path)
        x_test.append(img)
        length_test.append(int(lables[0],10))
        width_test.append(int(lables[1],10))
        angles_one_hot_test = np.eye(12)[int(lables[2],10)]
        angles_test.append(angles_one_hot_test)
        color_test.append(int(lables[3],10))

np.save('/home/dhanunjaya/Downloads/data/x_train.npy', x_train) 
np.save('/home/dhanunjaya/Downloads/data/length_train.npy', length_train)
np.save('/home/dhanunjaya/Downloads/data/width_train.npy', width_train)
np.save('/home/dhanunjaya/Downloads/data/angles_train.npy', angles_train)
np.save('/home/dhanunjaya/Downloads/data/color_train.npy', color_train)
np.save('/home/dhanunjaya/Downloads/data/x_test.npy', x_test)
np.save('/home/dhanunjaya/Downloads/data/length_test.npy', length_test)
np.save('/home/dhanunjaya/Downloads/data/width_test.npy', width_test)
np.save('/home/dhanunjaya/Downloads/data/angles_test.npy', angles_test)
np.save('/home/dhanunjaya/Downloads/data/color_test.npy', color_test)    
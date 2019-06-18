import os
import numpy as np
import shutil
import cv2
import re

def read_data():

    root_dir = os.getcwd()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    train_files = os.listdir('./train/')
    test_files = os.listdir('./test/')

    for classes in train_files:
        class_num = re.sub('.*?([0-9]*)$',r'\1',classes)
        imgs = os.listdir('./train/' + classes)
        for img in imgs:
            imgPath = './train/'+classes+'/'+img
            X_train.append(cv2.imread(imgPath))
            y_temp = np.eye(96)[class_num]
            Y_train.append(y_temp)


    for classes in test_files:
        class_num = re.sub('.*?([0-9]*)$',r'\1',classes)
        imgs = os.listdir('./test/' + classes)
        for img in imgs:
            imgPath = './test/'+classes+'/'+img
            X_test.append(cv2.imread(imgPath))
            y_temp = np.eye(96)[class_num]
            Y_test.append(y_temp)
    
    return (X_train, Y_train, X_test, Y_test)

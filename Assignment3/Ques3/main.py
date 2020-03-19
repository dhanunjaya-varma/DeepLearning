#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:11:40 2019

@author: dhanunjaya
"""

import numpy as np
import cv2
from keras.models import load_model

#img_test=np.load('./img_test.npy')
#mask_test=np.load('./mask_test.npy')

#mask_test = np.reshape(mask_test, (len(mask_test), 256, 256, 1))

#mask_test = mask_test.astype('float32') / 255
#img_test = img_test.astype('float32') / 255

model=load_model('./core_point.h5')
input_img=cv2.imread('./new/0276_3.jpeg')
orgX=input_img.shape[0]
orgY=input_img.shape[1]
#img=cv2.resize(input_img,(256,256))
img_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
thresh = np.expand_dims(thresh, axis=2)
img=cv2.resize(thresh,(256,256))
img = np.expand_dims(img, axis = 0)
img = np.expand_dims(img, axis = 3)
img = img.astype('float32') / 255
pred = model.predict(img)
pred = pred[0]*256
pred=np.squeeze(pred, axis=2)
cv2.imwrite('./new/0276_3_mask.jpeg', pred)
mask=cv2.resize(pred,(orgY,orgX))
#mask=cv2.resize(mask,(orgY,orgX))
cv2.imwrite('./new/0276_3_mask_org.jpeg', mask)
M = cv2.moments(mask)
cX_new = int(M["m10"] / M["m00"])
cY_new = int(M["m01"] / M["m00"])
M = cv2.moments(pred)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
cX=(cX/256)*orgY
cY=(cY/256)*orgX


print(cX_new)
print(cY_new)
print(cX)
print(cY)

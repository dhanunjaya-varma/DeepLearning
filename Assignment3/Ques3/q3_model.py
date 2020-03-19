#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:43:52 2019

@author: dhanunjaya
"""

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Conv2DTranspose, concatenate, add 
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

img_train=np.load('./img_train.npy')
mask_train=np.load('./mask_train.npy')
img_test=np.load('./img_test.npy')
mask_test=np.load('./mask_test.npy')

#mask_train = np.reshape(mask_train_org, (len(mask_train), 256, 256, 1))
#mask_test = np.reshape(mask_test_org, (len(mask_test), 256, 256, 1))
#img_train = np.reshape(img_train_org, (len(img_train), 256, 256, 1))
#img_test = np.reshape(img_test_org, (len(img_test), 256, 256, 1))

mask_train = np.expand_dims(mask_train, axis = 3)
mask_test = np.expand_dims(mask_test, axis = 3)
img_train = np.expand_dims(img_train, axis = 3)
img_test = np.expand_dims(img_test, axis = 3)


img_train = img_train.astype('float32') / 255
img_test = img_test.astype('float32') / 255
mask_train = mask_train.astype('float32') / 255
mask_test = mask_test.astype('float32') / 255

img_test, img_val, mask_test, mask_val = train_test_split(img_test, mask_test, test_size=0.4)

n_filters=16
dropout=0.5
batchnorm=True

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

input_img = Input((256, 256, 1), name='img')
c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
p1 = MaxPooling2D((2, 2)) (c1)
p1 = Dropout(dropout*0.5)(p1)

c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
p2 = MaxPooling2D((2, 2)) (c2)
p2 = Dropout(dropout)(p2)

c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
p3 = MaxPooling2D((2, 2)) (c3)
p3 = Dropout(dropout)(p3)

c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
p4 = Dropout(dropout)(p4)
    
c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
# expansive path
u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
u6 = Dropout(dropout)(u6)
c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
u7 = Dropout(dropout)(u7)
c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
u8 = Dropout(dropout)(u8)
c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
u9 = Dropout(dropout)(u9)
c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[input_img], outputs=[outputs])

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
#model.summary()

results = model.fit(img_train, mask_train, batch_size=32, epochs=8, 
                    validation_data=(img_val, mask_val))
model.save('./core_point.h5')

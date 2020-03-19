#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:24:22 2019

@author: dhanunjaya
"""

import matplotlib.pyplot as plt
from keras import backend as K
import keras
import cv2
import numpy as np

def heatmap_linedata(line_img):
    model = keras.models.load_model('/home/dhanunjaya/Downloads/q2/MultiHead_LineData')
    img_tensor = np.expand_dims(line_img, axis=0)
    preds = model.predict(img_tensor)
    X=np.argmax(preds[3])
    p=model.output[3][:,X]
    last_conv_layer = model.get_layer('conv2d_2')
    grads = K.gradients(p, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value=iterate([img_tensor])
    for i in range(32):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #fig = plt.figure(figsize=(12,8))
    #ax = fig.add_subplot(1,2,1)
    #ax.imshow(line_img)
    #ax = fig.add_subplot(1,2,2)
    #ax.imshow(heatmap)
    plt.matshow(heatmap)
    #cv2.imwrite('/home/dhanunjaya/Downloads/Assignment2/line_img.jpeg', line_img)
    #heatmap = cv2.resize(heatmap, (line_img.shape[1], line_img.shape[0]))
    #heatmap = np.uint8(255 * heatmap)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    #superimposed_img = heatmap.reshape(28,28,1) + X_train[65]
    #plt.matshow(np.squeeze(superimposed_img), cmap='gray')
    ##superimposed_img = heatmap + line_img
    ##cv2.imwrite('/home/dhanunjaya/Downloads/Assignment2/superimposed_img.jpeg', superimposed_img)
    ##ax = fig.add_subplot(1,3,3)
    ##ax.imshow(superimposed_img)

def heatmap_mnist(mnist_img):
    model = keras.models.load_model('/home/dhanunjaya/Downloads/q2/mnist.h5')
    img_tensor = np.expand_dims(mnist_img, axis=0)
    preds = model.predict(img_tensor)
    X=np.argmax(preds)
    p=model.output[:,X]
    last_conv_layer = model.get_layer('conv2d_4')
    grads = K.gradients(p, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value=iterate([img_tensor])
    for i in range(32):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,2,1)
    ax.imshow(mnist_img.reshape(28,28),cmap='gray')
    ax = fig.add_subplot(1,2,2)
    ax.imshow(heatmap)
    #plt.matshow(heatmap)
    #cv2.imwrite('/home/dhanunjaya/Downloads/Assignment2/line_img.jpeg', line_img)
    #heatmap = cv2.resize(heatmap, (28, 28))
    #heatmap = np.uint8(255 * heatmap)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #superimposed_img = heatmap + mnist_img
    #cv2.imwrite('/home/dhanunjaya/Downloads/Assignment2/superimposed_img.jpeg', superimposed_img)
    #ax = fig.add_subplot(1,3,3)
    #ax.imshow(superimposed_img.reshape(28,28),cmap='gray')

img1='/home/dhanunjaya/Downloads/Assignment1/Class47/1_1_5_0_15.jpeg'
heatmap_linedata(cv2.imread(img1))
img2='/home/dhanunjaya/Downloads/Assignment1/Class49/0_0_6_0_158.jpeg'
heatmap_linedata(cv2.imread(img2))
img3='/home/dhanunjaya/Downloads/Assignment1/Class56/1_1_6_1_8.jpeg'
heatmap_linedata(cv2.imread(img3))


from keras.datasets import mnist

# Data loading + reshape to 4D
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255
heatmap_mnist(X_train[65])
heatmap_mnist(X_train[59])
heatmap_mnist(X_train[29])
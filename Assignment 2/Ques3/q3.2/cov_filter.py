#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:03:43 2019

@author: dhanunjaya
"""


import matplotlib.pyplot as plt
from keras import backend as K
import keras
import numpy as np

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1 
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model, layer_name, filter_index, size):
    # Build a loss function that maximizes the activation
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]
    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def generate_pattern_grey(model, layer_name, filter_index, size):
    # Build a loss function that maximizes the activation
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]
    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.
    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def conv_filters_line_data():
    model = keras.models.load_model('./MultiHead_LineData')
    layer_name = 'conv2d_2'
    size = 28
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(32):
        ax = fig.add_subplot(6,6,i+1)
        ax.imshow(generate_pattern(model, layer_name, i , size=size))
        
def conv_filters_mnist():
    model = keras.models.load_model('./mnist.h5')
    layer_name = 'conv2d_4'
    size = 28
    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(32):
        ax1 = fig.add_subplot(6,6,i+1)
        img1=generate_pattern_grey(model, layer_name, i , size=size)
        img1=img1.reshape(28,28)
        ax1.imshow(img1,cmap='gray')
        
conv_filters_line_data()
conv_filters_mnist()
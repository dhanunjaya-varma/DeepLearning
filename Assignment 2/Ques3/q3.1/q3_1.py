#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:26:45 2019

@author: dhanunjaya
"""

import matplotlib.pyplot as plt
import keras
import numpy as np
import cv2


def layer_activation(line_img, layer):
    model = keras.models.load_model('./MultiHead_LineData')
    img_tensor = np.expand_dims(line_img, axis=0)
    layer_outputs = [layer.output for layer in model.layers[1:6]]
    activation_model = keras.models.Model(model.input,layer_outputs)
    activations = activation_model.predict(img_tensor)
    layer_activation = activations[layer]
    for i in range(layer_activation.shape[3]):
        plt.matshow(layer_activation[0, :, :, i], cmap='viridis')
    plt.show()
    

img='./1_1_5_0_75.jpeg'
layer_activation(cv2.imread(img),0)   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:29:41 2019

@author: dhanunjaya
"""


import numpy as np
from keras import layers
from keras import Input
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

#/home/dhanunjaya/Downloads/data
x_train=np.load('/home/dhanunjaya/Downloads/data/x_train.npy')
length_train=np.load('/home/dhanunjaya/Downloads/data/length_train.npy')
width_train=np.load('/home/dhanunjaya/Downloads/data/width_train.npy')
color_train=np.load('/home/dhanunjaya/Downloads/data/color_train.npy')
rotation_train=np.load('/home/dhanunjaya/Downloads/data/angles_train.npy')
x_test=np.load('/home/dhanunjaya/Downloads/data/x_test.npy')
length_test=np.load('/home/dhanunjaya/Downloads/data/length_test.npy')
width_test=np.load('/home/dhanunjaya/Downloads/data/width_test.npy')
color_test=np.load('/home/dhanunjaya/Downloads/data/color_test.npy')
rotation_test=np.load('/home/dhanunjaya/Downloads/data/angles_test.npy')

lines_input = Input(shape=(28,28,3), name='line')
x = layers.Conv2D(32, (7, 7), activation='relu')(lines_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2),strides=2)(x)
x = layers.Flatten()(x)

length_classification = layers.Dense(256,activation='relu')(x)
length_classification = layers.Dense(1, activation='sigmoid', name='length')(length_classification)

width_classification = layers.Dense(256,activation='relu')(x)
width_classification = layers.Dense(1, activation='sigmoid', name='width')(width_classification)

color_classification = layers.Dense(256,activation='relu')(x)
color_classification = layers.Dense(1, activation='sigmoid', name='color')(color_classification)

rotation_classification = layers.Dense(256,activation='relu')(x)
rotation_classification = layers.Dense(12, activation='softmax', name='rotation')(rotation_classification)

model = Model(lines_input, [length_classification, width_classification, color_classification,rotation_classification])


model.compile(optimizer='adam',
              loss={'length': 'binary_crossentropy',
                    'width': 'binary_crossentropy',
                    'color': 'binary_crossentropy',
                    'rotation': 'categorical_crossentropy'},
              metrics=["accuracy"])

history = model.fit(x_train, [length_train, width_train, color_train,rotation_train], 
                    epochs=4, 
                    batch_size=128,
                    verbose=1, 
                    validation_data=(x_test, [length_test, width_test, color_test,rotation_test]))

results = model.evaluate(x_test, [length_test, width_test, color_test,rotation_test])


lossNames = ["loss", "length_loss", "width_loss","color_loss","rotation_loss"]
plt.style.use("ggplot")
(fig, axis) = plt.subplots(5, 1, figsize=(28, 28))
EPOCHS=4

for (i, l) in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    axis[i].set_title(title)
    axis[i].set_xlabel("Epoch #")
    axis[i].set_ylabel("Loss")
    axis[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
    axis[i].plot(np.arange(0, EPOCHS), history.history["val_" + l],
               label="val_" + l)
    axis[i].legend()
 

plt.tight_layout()
plt.savefig("{}_losses.png".format('/home/dhanunjaya/Downloads/data'))
plt.close()

accuracyNames = ["length_acc", "width_acc","color_acc","rotation_acc"]
plt.style.use("ggplot")
(fig, axis) = plt.subplots(4, 1, figsize=(28, 28))
 

for (i, l) in enumerate(accuracyNames):
    axis[i].set_title("Accuracy for {}".format(l))
    axis[i].set_xlabel("Epoch #")
    axis[i].set_ylabel("Accuracy")
    axis[i].plot(np.arange(0, EPOCHS), history.history[l], label=l)
    axis[i].plot(np.arange(0, EPOCHS), history.history["val_" + l],
                 label="val_" + l)
    axis[i].legend()
 
plt.tight_layout()
plt.savefig("{}_accs.png".format('/home/dhanunjaya/Downloads/data'))
plt.close()


predictions = model.predict(x_test)
#Confusion matrix for all 4 output heads
length_matrix = confusion_matrix(length_test.astype(np.int), np.ceil(predictions[0]).astype(np.int).reshape(-1))
width_matrix = confusion_matrix(width_test.astype(np.int), np.ceil(predictions[1]).astype(np.int).reshape(-1))
color_matrix = confusion_matrix(color_test.astype(np.int), np.ceil(predictions[2]).astype(np.int).reshape(-1))
rotation_matrix = confusion_matrix(rotation_test.argmax(axis=1), predictions[3].argmax(axis=1))

print ("length head confusion matrix:")
print (length_matrix)

print ("width head confusion matrix:")
print (width_matrix)

print ("color head confusion matrix:")
print (color_matrix)

print ("rotation head confusion matrix:")
print (rotation_matrix)

print ("#####################################################################")
#precision_recall_fscore_support for all 4 heads
length_params = precision_recall_fscore_support(length_test.astype(np.int), np.ceil(predictions[0]).astype(np.int).reshape(-1))
width_params = precision_recall_fscore_support(width_test.astype(np.int), np.ceil(predictions[1]).astype(np.int).reshape(-1))
color_params = precision_recall_fscore_support(color_test.astype(np.int), np.ceil(predictions[2]).astype(np.int).reshape(-1))
rotation_params = precision_recall_fscore_support(rotation_test.argmax(axis=1), predictions[3].argmax(axis=1))
print ("length head params:")
print (length_params)

print ("width head fscores:")
print (width_params)

print ("color head fscores:")
print (color_params)

print ("rotation head fscores:")
print (rotation_params)

#model evaluation result

print('Test total loss:', results[0])
print('Test length loss:', results[1])
print('Test width loss:', results[2])
print('Test color loss:', results[3])
print('Test rotation loss:', results[4])
print('Test length accuracy:', results[5])
print('Test width accuracy:', results[6])
print('Test color accuracy:', results[7])
print('Test rotation accuracy:', results[8])

model.save('MultiHead_LineData.h5')
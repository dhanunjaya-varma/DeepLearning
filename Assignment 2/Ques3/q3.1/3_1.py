
# coding: utf-8

# In[4]:



import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras import backend as K
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

X_train.shape

number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

model = Sequential()
model.add(Convolution2D(32, (7, 7), input_shape=(1,28,28), data_format='channels_first'))
out1 = Activation('relu')
model.add(out1)
model.add(BatchNormalization())
out2 = MaxPooling2D(pool_size=(2, 2), strides=2)
model.add(out2)
model.add(Flatten())
model.add(Dense(1024))
out3 = Activation('relu')
model.add(out3)
model.add(Dense(number_of_classes))
model.add(Activation('softmax'))


img_to_visualize = X_train[3]
img_to_visualize = np.expand_dims(img_to_visualize, axis=0)
plt.imshow(np.squeeze(X_train[3]), cmap='gray')
plt.show()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=3, validation_data=(X_test, Y_test))



# function to visualise intermediate activation layers
def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))
    
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')
    plt.show()


layer_to_visualize(out1)    
# layer_to_visualize(out2)
# layer_to_visualize(out3)


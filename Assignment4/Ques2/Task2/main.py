import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from keras.callbacks import ModelCheckpoint
import random
from sklearn.model_selection import train_test_split


def generate_training_list(leng):
    training_list = []

    for l in range(leng):
        common_diff = random.randint(1, 5)
        length = random.randint(2, 10)
        seq = []
        start_ele = random.randint(1, 100)

        for n in range(length+1):
            ele = start_ele + (n-1)*common_diff
            if (ele >= 0):
                seq.append(ele)
        last = seq[-1]
        
        seq.pop()
        seq += [0]*(10 - len(seq))
        
        # print (seq, last)
        training_list.append((seq, last))

    return training_list


training_list = generate_training_list(10000)
X = [x[0] for x in training_list]
Y = [x[1] for x in training_list]
X = np.asarray(X)
Y = np.asarray(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
X_train = np.expand_dims(X_train, 2)
X_test = np.expand_dims(X_test, 2)

Y_train = np.expand_dims(Y_train, 1)
Y_test = np.expand_dims(Y_test, 1)
# print (Y_test.shape)

print (X_train.shape)
print (Y_train.shape)

epochs = 250
batch_size = 64
hidden_neurons = 32
output_size = 1

model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(output_size, activation = 'elu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# filepath="ap_d.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(X_test, Y_test))



filepath="weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

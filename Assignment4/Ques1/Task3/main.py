import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input, RepeatVector
import numpy as np
import random
from sklearn.model_selection import train_test_split

def f_n(n):
    if n%2:
        return 3*n+1
    else:
        return int(n/2)

def iter_count(start_ele):
    count = 0
    while (start_ele > 1):
        count += 1
        start_ele = f_n(start_ele)
    return count

def generate_training_list(leng):
    training_list = []

    for l in range(leng):
        seq = []
        start_ele = random.randint(10, 1000)

        while (start_ele > 1):
            seq.append(start_ele)
            start_ele = f_n(start_ele)

        first = seq[0]
        seq.append(1)
        seq.pop(0)        
        input_seq = []
        input_seq.append(first)
        # input_seq += [0]*(len(seq)-1)

        training_list.append((np.asarray(input_seq), seq))

    return training_list


training_list = generate_training_list(10)

X = [x[0] for x in training_list]
Y = [x[1] for x in training_list]
X = np.asarray(X)
Y = np.asarray(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
# X_train = np.expand_dims(X_train, 2)
# X_test = np.expand_dims(X_test, 2)

Y_train = np.expand_dims(Y_train, 1)
Y_test = np.expand_dims(Y_test, 1)
# print (Y_test.shape)

print (X_train.shape)
print (Y_train.shape)

print (X_train)
epochs = 100
batch_size = 64
hidden_neurons = 32
output_size = 1

model = Sequential()
model.add(RepeatVector(10, input_shape=(1, 1)))
model.add(LSTM(hidden_neurons, return_sequences=True))
model.add(Dense(output_size, activation = 'elu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
model.save('ap_'+str(common_diff)+'.h5')
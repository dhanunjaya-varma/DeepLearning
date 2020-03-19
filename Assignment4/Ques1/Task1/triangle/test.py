import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import random
import collections
from collections import deque 
from sklearn.model_selection import train_test_split
from keras.models import load_model
from scipy import signal
import matplotlib.pyplot as plt

model = load_model('triangle.h5')
print (model.summary())

t = np.linspace(0, 1, 1000)
triangle = signal.sawtooth(2 * np.pi * 5 * t)

plt.plot(t, triangle)
# plt.show()

start = [triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8], triangle[9]]

all_ele = []
temp = collections.deque(maxlen=10)
temp = start

for idx in range(10, len(t)):
    flag = 0    
    new_ele = triangle[idx]
    temp.append(new_ele)

    temp1 = deque(temp, maxlen=10)

    Xnew = np.asarray(temp1)
    Xnew = np.expand_dims(Xnew, 2)
    Xnew = np.expand_dims(Xnew, 0)
    # print (Xnew.shape)

    if((Xnew < 0).all()):
        flag = 1
        Xnew *= -1

    ynew = model.predict(Xnew)

    if flag == 1:
        ynew[0][0] *= -1

    all_ele.append(ynew[0][0])
    # print(ynew[0][0])

all_ele2 = [triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8], triangle[9]] + all_ele
print (len(triangle))
print (len(all_ele2))

plt.plot(t, all_ele2)

plt.show()



# a += [0]*(10 - len(a))
# Xnew = np.asarray(a)
# Xnew = np.expand_dims(Xnew, 2)
# Xnew = np.expand_dims(Xnew, 0)
# # print (Xnew.shape)
# ynew = model.predict(Xnew)
# print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

# print (triangle[114])
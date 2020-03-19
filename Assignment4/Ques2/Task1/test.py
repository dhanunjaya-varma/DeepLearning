import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import collections
from collections import deque 
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt

def is_negative(ls):
    for i in ls:
        if i < 0:
            return True
    return False

model = load_model('sin3.hdf5')
print (model.summary())

t = np.arange(0.0, 10.0, 0.001)
orig_sin = np.sin(t)
# print (orig_sin)

plt.plot(t, orig_sin)
# plt.show()


start = [np.sin(0e-3), np.sin(1e-3), np.sin(2e-3), np.sin(3e-3), np.sin(4e-3), np.sin(5e-3), np.sin(6e-3), np.sin(7e-3), np.sin(8e-3), np.sin(9e-3)]

all_ele = []
temp = collections.deque(maxlen=10)
temp = start


for idx in range(10, len(t)):

    flag = 0    
    new_ele = np.sin(t[idx])
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

all_ele2 = [np.sin(0e-3), np.sin(1e-3), np.sin(2e-3), np.sin(3e-3), np.sin(4e-3), np.sin(5e-3), np.sin(6e-3), np.sin(7e-3), np.sin(8e-3), np.sin(9e-3)] + all_ele
print (len(orig_sin))
print (len(all_ele2))

plt.plot(t, all_ele2)

plt.show()

# a += [0]*(10-len(a))
# Xnew = np.asarray(a)
# Xnew = np.expand_dims(Xnew, 2)
# Xnew = np.expand_dims(Xnew, 0)
# # print (Xnew.shape)
# ynew = model.predict(Xnew)
# print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

# print (np.sin(3e-3))
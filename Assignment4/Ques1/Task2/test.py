import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model

model_path = 'models/ap_1.h5'
# model_path = 'ap_1.hdf5'
model = load_model(model_path)
print (model.summary())

with open('ap_1.pkl', 'rb') as f:
    data = pd.load(f)

Xs = [[1,2,3,4,5,6,7]]
Ys = [x[1] for x in data]

for a in Xs:
    a += [0]*(10 - len(a))
    Xnew = np.asarray(a)
    Xnew = np.expand_dims(Xnew, 2)
    Xnew = np.expand_dims(Xnew, 0)
    # print (Xnew.shape)
    ynew = model.predict(Xnew)
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

# a = [1,6,11,16]
# a += [0]*(10 - len(a))
# Xnew = np.asarray(a)
# Xnew = np.expand_dims(Xnew, 2)
# Xnew = np.expand_dims(Xnew, 0)
# # print (Xnew.shape)
# ynew = model.predict(Xnew)
# print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
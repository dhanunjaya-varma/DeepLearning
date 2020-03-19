import pickle as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model

model = load_model('ap_5.h5')
print (model.summary())


a = [1,6,11,16]
a += [0]*(10 - len(a))
Xnew = np.asarray(a)
Xnew = np.expand_dims(Xnew, 2)
Xnew = np.expand_dims(Xnew, 0)
# print (Xnew.shape)
ynew = model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
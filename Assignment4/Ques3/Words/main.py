from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from tensorflow import set_random_seed
from numpy.random import seed
from keras.callbacks import ModelCheckpoint

import io
import pandas as pd
import numpy as np
import string, os 
import random
import sys
import gensim
from keras.layers import Dense, Activation
from keras.utils.data_utils import get_file

path = './pap.txt'

max_sentence_len = 40
with open(path) as file_:
  docs = file_.readlines()

# each sentence represented as list of words
sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:max_sentence_len]] for doc in docs]
print('Num sentences:', sentences[4])

# word2vec part
word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
# convert a one-hot encoding of a word into a dense embedding-vector of the right dimensionality
pretrained_weights = word_model.wv.syn0 # syn0 array essentially holds raw word-vectors

print (pretrained_weights.shape) # (8430, 100)
vocab_size, emdedding_size = pretrained_weights.shape

for word in ['she', 'her', 'they', 'home']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))

# utility functions
def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

# data for lstm
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  for t, word in enumerate(sentence[:-1]):
    train_x[i, t] = word2idx(word)
  train_y[i] = word2idx(sentence[-1])

print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

##################################
# model
##################################
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
##################################

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

filepath="gru_weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(train_x, train_y,
          batch_size=128,
          epochs=100,
          callbacks=callbacks_list)

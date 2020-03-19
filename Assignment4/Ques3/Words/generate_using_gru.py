from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
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
global str


from keras.models import load_model
model = load_model('gru_weights-improvement.hdf5')
print (model.summary())

# utility functions
def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]


#############################################################
path = './pap.txt'
max_sentence_len = 40
with open(path) as file_:
  docs = file_.readlines()

# each sentence represented as list of words
sentences = [[word for word in doc.lower().translate(string.punctuation).split()[:max_sentence_len]] for doc in docs]
print('Num sentences:', sentences[4])

word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
#############################################################

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

texts = [
    # 'Elizabeth wanted to run away in the distance because',
    # 'Mr. Darcy was a very',
    'Mrs. Bennet was so excited that she could',
    # 'Mr. Wickham, a tall and dashing young man, made',
    # 'Lady Catherine, having heard rumours about Elizabeth and Darcy, visits',
]

for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))
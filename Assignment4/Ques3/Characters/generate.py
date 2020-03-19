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
import io
import pandas as pd
import numpy as np
import string, os 
import random
import sys

from keras.models import load_model
model = load_model('text.h5')

# def generate_text(seed_text, next_words, model, max_sequence_len):
#     for _ in range(next_words):
#         token_list = tokenizer.texts_to_sequences([seed_text])[0]
#         token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
#         predicted = model.predict_classes(token_list, verbose=0)
        
#         output_word = ""
#         for word,index in tokenizer.word_index.items():
#             if index == predicted:
#                 output_word = word
#                 break
#         seed_text += " "+output_word
#     return seed_text.title()


path = 'pap.txt'
with io.open(path, encoding='utf-8') as f:
    text_orig = f.read().lower()

path = 'text_gen_test.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text_orig)))
char_indices = dict((c, i) for i, c in enumerate(chars)) # char to index mapping
indices_char = dict((i, c) for i, c in enumerate(chars)) # index to char mapping


generated = ''
maxlen = 40
# 52 20 41 47 70
sentence = text[0: maxlen]
generated += sentence
print('----- Generating with seed: "' + sentence + '"')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for i in range(400):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = indices_char[next_index]

    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
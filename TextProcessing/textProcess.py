# -*- coding: utf-8 -*-
"""LoadingInTextDataForNextWordPrediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HCgKn5XQ7Q3ywxGszVWx2kddfT9UBASp
"""

# modified from: https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer

import string

# changeable params
my_file = "allofjane.txt"
seq_len = 100

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
 
# turn a doc into clean tokens
def clean_doc(doc):
 # replace '--' with a space ' '
 doc = doc.replace('--', ' ')
 # split into tokens by white space
 tokens = doc.split()
 # remove punctuation from each token
 table = str.maketrans('', '', string.punctuation)
 tokens = [w.translate(table) for w in tokens]
 # remove remaining tokens that are not alphabetic
 tokens = [word for word in tokens if word.isalpha()]
 # make lower case
 tokens = [word.lower() for word in tokens]
 return tokens
 
# save tokens to file, one dialog per line
def save_doc(lines, filename):
 data = '\n'.join(lines)
 file = open(filename, 'w')
 file.write(data)
 file.close()
 
# load document
doc = load_doc(my_file)
print(doc[:200])
 
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
 
# organize into sequences of tokens
length = seq_len + 1
sequences = list()
for i in range(length, len(tokens)):
 # select sequence of tokens
 seq = tokens[i-length:i]
 # convert into a line
 line = ' '.join(seq)
 # store
 sequences.append(line)
print('Total Sequences: %d' % len(sequences))
 
# save sequences to file
out_filename = my_file[:-4] + '_seq.txt'
save_doc(sequences, out_filename)

# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
 
# load
doc = load_doc(out_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1


# separate into input and output
sequences = np.array(sequences)
sequences.shape
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

p_train = 0.8

n_train = int(X.shape[0]//(1/p_train))
X_train = X[0:n_train]
y_train = y[0:n_train]
X_test = X[n_train:]
y_test = y[n_train:]

X_train.shape

X_test.shape

y_train

y_test

y_train.shape

y_test.shape
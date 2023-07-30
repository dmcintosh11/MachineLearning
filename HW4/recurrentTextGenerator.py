
# modified from: https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.text import Tokenizer

import string

# changeable params
my_file = "gatsby.txt"
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

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(150))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history1 = model.fit(X_train, y_train, batch_size=128, epochs=50)


pd.DataFrame(history1.history)[['loss','accuracy']].plot()
plt.title("Accuracy and Loss Model 1")
plt.savefig('../HW4/Mod1.png')


model2 = Sequential()
model2.add(Embedding(vocab_size, 50, input_length=seq_length))
model2.add(LSTM(100, return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(100, return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(150))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(vocab_size, activation='softmax'))
print(model2.summary())

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history2 = model2.fit(X_train, y_train, batch_size=128, epochs=50)


pd.DataFrame(history2.history)[['loss','accuracy']].plot()
plt.title("Accuracy and Loss Model 2")
plt.savefig('../HW4/Mod2.png')

from random import randint
from pickle import load
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# load doc into memory
def load_doc(filename):
 # open the file as read only
 file = open(filename, 'r')
 # read all text
 text = file.read()
 # close the file
 file.close()
 return text
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
  result = list()
  in_text = seed_text
# generate a fixed number of words
    for _ in range(n_words):
# encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
# truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # predict probabilities for each word
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        #print('worrs #: ' + str(_) + str(yhat))
    # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
            #print('word #: ' + str(_) + 'outword: ' + out_word)
                break
# append to input
        in_text += ' ' + out_word
    result.append(out_word)
    return ' '.join(result)
 
# load cleaned text sequences
in_filename = 'gatsby_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

for i in range(1,11):
# select a seed text
    seed_text = lines[randint(0,len(lines))]
    #print(seed_text + '\n')
    
    # generate new text
    generated = generate_seq(model, tokenizer, seq_length, seed_text, 25)
    print('Original Model Text ' + str(i) + ': ')
    print('SEED text: ' + seed_text + '\n')
    print('GENERATED text: ' + generated)
    print('--------------------------')
  
for i in range(1,11):
# select a seed text
    seed_text = lines[randint(0,len(lines))]
    
    # generate new text
    generated = generate_seq(model2, tokenizer, seq_length, seed_text, 25)
    print('Deep LSTM Model Text: ' + str(i) + ': ')
    print('SEED text: ' + seed_text + '\n')
    print('GENERATED text: ' + generated)
    print('--------------------------')
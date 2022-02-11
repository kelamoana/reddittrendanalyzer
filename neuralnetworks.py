# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import helpers

if __name__ == "__main__":

    print("Testing RNN")

    # Data
    sent_and_classes = helpers.get_sentences_and_classes()
    word_embed_dict = helpers.convert_to_wordemb([sent for sent in sent_and_classes])

    # RNN instance
    embed_dim = 100
    lstm_out = 1
    max_features = len(word_embed_dict)

    model = Sequential()
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    # Get x_train and y_train
    x_train = np.asarray([embed for embed in word_embed_dict.values()])
    y_train = np.array([int(v) for v in sent_and_classes.values()])

    model.fit(x_train, y_train)



    print("Testing CNN")

    # CNN instance
    input_shape = (len(word_embed_dict), 15, 100)

    model = Sequential()
    model.add(Conv1D(1, 15, activation="relu", input_shape=input_shape[1:]))
    model.add(MaxPooling1D(pool_size=1))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

    model.fit(x_train, y_train)


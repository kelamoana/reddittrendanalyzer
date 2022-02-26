from sys import set_asyncgen_hooks
from tokenize import Token
import gensim
import gensim.downloader as api
import nltk
import re
import pandas as pd
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU, Conv1D, MaxPooling1D, Flatten
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api

stopwords = set(nltk.corpus.stopwords.words('english'))


def filter_stopwords(sentence):
    """
    Removes stop words from a sentence.

    Parameters: Sentence - a string of text.
    Returns: A list of words with no punctuation or stop english words.
    """

    # Split sentence into words
    word_list = sentence.lower().split()

    # Remove punc and add word for each non-stop word
    return [re.sub(r'[^\w\s]', '', w) for w in word_list if w not in stopwords]

def get_sentences_and_classes(corpus = "data/logisticRegression/XBinary.txt", classes = "data/logisticRegression/YBinary.txt"):
    """
    A function to process all of the training data and turn it
    into a usable data structure.

    Parameters: corpus - A txt file, classes - a txt file
    Returns: A list where each val is a tuple containing
    the text and a list of its sentiments.
    """

    # Create File Obj
    file_obj1 = open(corpus, "r", encoding='utf-8', errors='ignore')
    file_obj2 = open(classes, "r", encoding='utf-8', errors='ignore')

    # List to return
    file_lines = []

    # Grab the first line
    line1 = file_obj1.readline()
    line2 = file_obj2.readline()

    # While we haven't reached the end
    while line1 != '':

        # Add to dict
        file_lines.append( [line1.replace("\n", ''), line2.replace("\n", '')] )

        # Grab next line
        line1 = file_obj1.readline()
        line2 = file_obj2.readline()

    return file_lines

def convert_to_list_of_words(sentences_with_classes):
    lstOfSentencesOfWords = []
    for sentence in sentences_with_classes:
        # we only care about the keys here
        lstOfWords = filter_stopwords(sentence)
        lstOfSentencesOfWords.append(lstOfWords)
    return lstOfSentencesOfWords

if __name__ == "__main__":       
    # # Grabbing Data File and turning into CSV
    # folder = 'aclImdb'
    # labels = {'pos': 1, 'neg': 0}
    # df = pd.DataFrame()
    # for f in ('test', 'train'):    
    #     for l in ('pos', 'neg'):
    #         path = os.path.join(folder, f, l)
    #         for file in os.listdir (path) :
    #             with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
    #                 txt = infile.read()
    #             df = df.append([[txt, labels[l]]],ignore_index=True)
    # df.columns = ['review', 'sentiment']
    # df.to_csv("movie_data.csv", index=False, encoding="utf-8")

    # Grab data from CSV to split
    # df = pd.DataFrame()
    # df = pd.read_csv('movie_data.csv', encoding='utf-8')
    # df.head(3)

    # # Split Data Manually
    # X_train = df.loc[:24999, 'review'].values
    # Y_train = df.loc[:24999, 'sentiment'].values

    # X_test = df.loc[:25000, 'review'].values
    # Y_test = df.loc[:25000, 'sentiment'].values


    lines = get_sentences_and_classes()
    sentiments = np.asarray([float(line[1]) for line in lines])
    review_lines = np.asarray(convert_to_list_of_words([line[0] for line in lines]))

    '''for sentence in lines:
        # we only care about the keys here
        lstOfWords = filter_stopwords(sentence)
        review_lines.append(lstOfWords) '''

    EMBEDDING_DIM = 100

    print("Build model...")
    #corpus = api.load('text8')

    # Word Vector model
    # model = gensim.models.Word2Vec(sentences=review_lines, vector_size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
    # #filename = "text8Corpus_word2vec.txt"
    # filename = "reddit_embedding_word2vec.txt"
    # model.wv.save_word2vec_format(filename, binary=False)

    corpus = api.load('text8')
    model = gensim.models.Word2Vec(corpus)
    filename = "reddit_embedding_word2vec.txt"
    model.wv.save_word2vec_format(filename, binary=False)

    embeddings_index = {}
    word2vecFileObj = open(filename, encoding='utf-8')
    for line in word2vecFileObj:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    word2vecFileObj.close()

    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(review_lines)
    sequences = tokenizer_obj.texts_to_sequences(review_lines)

    # Pad Sequences
    word_index = tokenizer_obj.word_index

    # Define Vocab Size
    vocab_size = len(tokenizer_obj.word_index)

    # Pad data, such that the len of each sequence == len of the largest sentence
    review_pad = pad_sequences(sequences, maxlen=25)
    sentiment = sentiments

    # Print the shape of the padded sequences   
    print("Shape of review tensor:", review_pad.shape)
    print("Shape of sentiment tensor:", sentiment.shape)

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(num_words)


    # Define RNN model
    model = Sequential()
    embedding_layer = Embedding(num_words,
                        EMBEDDING_DIM, 
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=25,
                        trainable=False)
    model.add(embedding_layer)
    #model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


    # Shuffle all data and prepare to split
    VALIDATION_SPLIT = 0.2
    indices = np.arange(review_pad.shape[0])
    np.random.shuffle(indices)
    review_pad = review_pad[indices]
    sentiment = sentiment[indices]
    num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

    # Split the data using the sequences created
    X_train_pad = review_pad[:-num_validation_samples]
    y_train = sentiment[:-num_validation_samples]
    X_test_pad = review_pad[-num_validation_samples:]
    y_test = sentiment[-num_validation_samples:]

    print("Training RNN")
    model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

    # CNN instance

    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(32, kernel_size=8, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

    print("Training CNN")
    model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)
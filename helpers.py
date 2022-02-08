# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen


from sys import set_asyncgen_hooks
import gensim
import nltk
import re

emotions = { i:w for i,w in
    enumerate(
    ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment","disapproval", "disgust",
    "embarrassment","excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"])
    }

emotionsCategories = {
"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
"negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
"ambiguous": ["realization", "surprise", "curiosity", "confusion"]
}

stopwords = set(nltk.corpus.stopwords.words('english'))

def preprocess_data_tsv(filename = "data/train.tsv"):
    """
    A function to process all of the training data and turn it
    into a usable data structure.

    Parameters: A TSV file
    Returns: A dict where each val is a tuple containing
    the text and a list of its sentiments.
    """

    # Create File Obj
    file_obj = open(filename, "r", encoding='utf-8', errors='ignore')

    # Dict to return
    file_lines = {}

    # Grab the first line
    line = file_obj.readline()

    # While we haven't reached the end
    while line != '':

        # Create list of [text, sentiments, id] and remove newlines
        line_to_append = line.replace("\n", '').split('\t')

        # Turn sentiments from str to list of ints
        line_to_append[1] = [int(i) for i in line_to_append[1].replace(' ', '').split(',')]

        # Add to dict
        file_lines[line_to_append[2]] = (filter_stopwords(line_to_append[0]), line_to_append[1])

        # Grab next line
        line = file_obj.readline()

    return file_lines

def get_sentences_and_classes(corpus = "data/logisticRegression/XTrainData.txt", classes = "data/logisticRegression/YTrainData.txt"):
    """
    A function to process all of the training data and turn it
    into a usable data structure.

    Parameters: corpus - A txt file, classes - a txt file
    Returns: A dict where each val is a tuple containing
    the text and a list of its sentiments.
    """

    # Create File Obj
    file_obj1 = open(corpus, "r", encoding='utf-8', errors='ignore')
    file_obj2 = open(classes, "r", encoding='utf-8', errors='ignore')

    # Dict to return
    file_lines = {}

    # Grab the first line
    line1 = file_obj1.readline()
    line2 = file_obj2.readline()

    # While we haven't reached the end
    while line1 != '':

        # Add to dict
        file_lines[line1] = line2

        # Grab next line
        line1 = file_obj1.readline()
        line2 = file_obj2.readline()

    return file_lines


def filter_stopwords(sentence):
    """
    Removes stop words from a sentence.

    Parameters: Sentence - a string of text.
    Returns: A list of words with no punctuation or stop english words.
    """

    # Split sentence into words
    word_list = sentence.split()

    # Remove punc and add word for each non-stop word
    return [re.sub(r'[^\w\s]', '', w) for w in word_list if w not in stopwords]

def convert_to_wordemb(prepocessed_data):
    """
    Converts preprocessed data into word embeddings

    Parameters: preprocessed_data - a dict of sentences
    Returns: A dict, where each sentence is a key and associated with
        a list of vectors representing each word.
    """
    
    # Initialize Word2Vec model
    model = gensim.models.Word2Vec(prepocessed_data)

    # Leave this commented out for now 
    # model.train(prepocessed_data, total_words=10000, epochs=1)
   
    # Build the vocab of the model
    model.build_vocab(prepocessed_data)

    # FOR DEBUG
    # print([i for i in model.wv.index_to_key])

    embedded_word_list = dict()

    for s in prepocessed_data:
        
        embedded_word_list[s] = [model.wv[word] for word in s if word in model.wv]

    return embedded_word_list

if __name__ == "__main__":

    # data_dict = preprocess_data_tsv()
    # word_embed_list = convert_to_wordemb([text for text, sent in data_dict.values()])
    
    sent_and_classes = get_sentences_and_classes()
    word_embed_dict = convert_to_wordemb([sent for sent in sent_and_classes])

    i = 0
    for s in sent_and_classes:
        print(f"Sentence: {s} -------> Class: {sent_and_classes[s]}")

        if i == 1:
            break

        i+=1


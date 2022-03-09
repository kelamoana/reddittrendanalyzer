# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen


from operator import neg
from posixpath import split
from sys import set_asyncgen_hooks
import gensim
import gensim.downloader as api
import nltk
import re
import numpy as np


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
        file_lines[line1.replace("\n", '')] = line2.replace("\n", '')

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
    word_list = sentence.lower().split()

    # Remove punc and add word for each non-stop word
    return [re.sub(r'[^\w\s]', '', w) for w in word_list if w not in stopwords]

def convert_to_wordemb(prepocessed_data, sent_seq_size = 15):
    """
    Converts preprocessed data into word embeddings

    Parameters: preprocessed_data - a dict of sentences
    Returns: A dict, where each sentence is a key and associated with
        a list of vectors representing each word.
    """
    
    # # Initialize Word2Vec model
    # model = gensim.models.Word2Vec(prepocessed_data)

    # model.train(prepocessed_data, total_examples=len(prepocessed_data), epochs=10)
   
    # # Build the vocab of the model
    # # model.build_vocab(prepocessed_data)

    corpus = api.load('text8')
    model = gensim.models.Word2Vec(corpus)
    
    embedded_word_dict = dict()
    
    # print(type(model.wv['dog']))

    for sent, val in prepocessed_data.items():
        
        emb_sub_seq = np.ndarray((sent_seq_size, 100)) # [0 for i in range(sent_seq_size)]
        
        i = 0
        for word in filter_stopwords(sent):
            if i == sent_seq_size:
                break
            if word in model.wv:
                j = 0
                for num in model.wv[word]:
                    
                    emb_sub_seq[i][j] = num
                    j += 1

                i += 1

        embedded_word_dict[sent] = (emb_sub_seq, val)

    return embedded_word_dict

# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def convert_twitter_to_binary_classes(tsv_name, path, new_file_name):

    # Create File Obj
    file_to_read = open(tsv_name, "r", encoding='utf-8', errors='ignore')
    file_x = open(path + "X" + new_file_name, "w", encoding='utf-8', errors='ignore')
    file_y = open(path + "Y" + new_file_name, "w", encoding='utf-8', errors='ignore')
    # Grab the first line
    line = file_to_read.readline()

    # Skip the headers of the file
    line = file_to_read.readline()

    # Create Count to track Ratio of Pos to Neg
    pos_count = 1
    neg_count = 1

    # While we haven't reached the end
    while line != '':

        # Create list of [text, sentiments, id] and remove newlines
        split_line = line.split('\t')

        #Change Sentiment to 0 or 1
        if int(split_line[1]) in {1, 2}:
            
            split_line[1] = "1"

            if pos_count <= neg_count:
                # Write the new line to the file
                file_x.write(split_line[0] + '\n')
                file_y.write(split_line[1] + '\n')
                pos_count += 1
        else:
            split_line[1] = "0"

            if neg_count <= pos_count:
                # Write the new line to the file
                file_x.write(split_line[0] + '\n')
                file_y.write(split_line[1] + '\n')
                neg_count += 1

        # Grab next line
        line = file_to_read.readline()

    print(f"POS COUNT: {pos_count} and NEG COUNT: {neg_count}. RATIO: \
        {pos_count/(pos_count+neg_count)}: \
        {neg_count/(pos_count+neg_count)}")

if __name__ == "__main__":

    # data_dict = preprocess_data_tsv()
    # word_embed_list = convert_to_wordemb([text for text, sent in data_dict.values()])
    
    # sent_and_classes = get_sentences_and_classes()
    # word_embed_dict = convert_to_wordemb(sent_and_classes)

    # i = 0
    # for s in sent_and_classes:
    #     print(f"Sentence: {s} -------> Class: {sent_and_classes[s]}")
    #     print(f"Embed: {word_embed_dict[s]}")
    #     if i == 1:
    #         break

    #     i+=1

    print("Executing Main in Helpers.py...")

    convert_twitter_to_binary_classes("data/Twitter/test.txt", "data/Twitter/", "testBinary.txt")
    convert_twitter_to_binary_classes("data/Twitter/training.txt", "data/Twitter/", "trainingBinary.txt")
    convert_twitter_to_binary_classes("data/Twitter/validation.txt", "data/Twitter/", "validationBinary.txt")
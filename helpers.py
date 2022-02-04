# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

import gensim


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
        file_lines[line_to_append[2]] = (line_to_append[0], line_to_append[1])

        # Grab next line
        line = file_obj.readline()

    return file_lines

def convert_to_wordemb(prerpocessed_data):

    gensim.models.Word2Vec(prerpocessed_data)

    return None

if __name__ == "__main__":

    data_dict = preprocess_data_tsv()
    convert_to_wordemb([text for text, sent in data_dict.values()])

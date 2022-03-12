import numpy as np
from logisticRegression import train_bow_and_tfidf, create_logreg_matrices, reduceVocab
from lexiconanalyzer import vader_sentiment_score
from recurringNeuralNetworks import train_rnn_model, convert_to_list_of_words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

READ_PATH = 'data/RedditRealtime'
WRITE_PATH = 'data/RedditPredictions'
MONTHS = ['Dec', 'Jan', 'Feb', 'July']
SUBREDDITS = ['Russia', 'Ukraine', 'UCI']


def write_predictions(predictions, model, month, subreddit):
    """
    A generalized function for writing predictions to a file.

    Paramaters: Month and Subreddit that is predicted.
    Returns: None. Writes predictions to a file.
    """
    
    file_to_write = open(f"{WRITE_PATH}/{model}/{month}/{subreddit}.txt", 'w', encoding='utf-8', errors='ignore')

    for prediction in predictions:
        file_to_write.write(str(prediction) + '\n')

    file_to_write.close()


def get_lexicon_predictions(real_data_file):
    """
    A function that gets predictions from the lexicon analyzer.

    Parameters: File that contains real reddit data.
    Returns: List of predictions.
    """

    file_obj = open(real_data_file, 'r', encoding='utf-8', errors='ignore')
    predictions = list()

    lines = file_obj.readlines()

    for line in lines:        
        predictions.append(vader_sentiment_score(line))

    file_obj.close()

    return predictions

def create_predictions(get_pred_func, write_pred_func, model):
    """
    A generalized function that creates predictions.

    Parameters: A function that returns a list of predictions, a function that writes predictions, model name as a string.
    Returns: None.
    """

    for month in MONTHS:
        
        for subreddit in SUBREDDITS:
            
            # Write predictions to the files, after attaining them.
            write_pred_func(get_pred_func(f"{READ_PATH}/{month}/{subreddit}.txt"), model, month, subreddit)


def thresh(prediction):
    """
    Threshold function that returns 0 or 1 for the RNN prediction.

    Parameter: List of 1 value.
    Returns: None.
    """

    return 0 if prediction[0] < 0.5 else 1


def get_rnn_predictions(real_data_file):
    """
    Attains predictions from the RNN model.

    Parameters: Path to a file containing Reddit Data
    Returns: List of predictions
    """
    # Access the model
    # Used global to allow traning to occur only once
    global rnn_model

    # Open File
    file_obj = open(real_data_file, 'r', encoding='utf-8', errors='ignore')

    # Grab the lines
    lines = file_obj.readlines()

    # Convert to list of words and np array
    lines_as_np_array = np.asarray(convert_to_list_of_words(lines))

    # Create the Tokenizer and tokenize for embedding layer
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(lines_as_np_array)
    sequences = tokenizer_obj.texts_to_sequences(lines_as_np_array)
    
    # Pad for embedding layer
    lines_pad = pad_sequences(sequences, maxlen=25)


    return [thresh(prediction) for prediction in rnn_model.predict(lines_pad)] # List of predictions


def get_bow_predictions(real_data_file):
    global bow_model

    # file_obj = open(real_data_file, 'w', encoding='utf-8', errors='ignore')
    # lines = file_obj.readlines()

    tfidf_matrix, doc_term_matrix, tokens_in_a_list = create_logreg_matrices(real_data_file)
    return bow_model.predict(doc_term_matrix)     


if __name__ == "__main__":
    print("Executing Metrics.py ...")
    
    # Run Lexicon Predictions:
    # create_predictions(get_lexicon_predictions, write_predictions, "lexiconanalyzer")

    # Run RNN Predictions:
    # rnn_model = train_rnn_model()
    # create_predictions(get_rnn_predictions, write_predictions, "rnn")

    # Run LogReg Predictions
    bow_model, tfidf_model = train_bow_and_tfidf()
    create_predictions(get_bow_predictions, write_predictions, "logregbow")
    
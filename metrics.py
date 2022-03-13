import numpy as np
from logisticRegression import train_bow_and_tfidf, create_logreg_matrices, reduceVocab
from lexiconanalyzer import vader_sentiment_score
from recurringNeuralNetworks import train_rnn_model, convert_to_list_of_words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

DATA_PATH = 'data/RedditRealtime'
METRICS_PATH = 'data/Metrics'
PREDICTIONS_PATH = 'data/RedditPredictions'
MONTHS = ['Jul', 'Dec', 'Jan', 'Feb',]
SUBREDDITS = ['Russia', 'Ukraine', 'UCI']
MODELS = ['lexiconanalyzer', 'logregbow', 'logregtfidf', 'rnn']
METRIC_TYPES = ['monthlyavgs', 'monthlyposnegtotals', 'weeklyavgs', 'weeklyposnegtotals']


def write_predictions(predictions, model, month, subreddit):
    """
    A generalized function for writing predictions to a file.

    Paramaters: Month and Subreddit that is predicted.
    Returns: None. Writes predictions to a file.
    """
    
    file_to_write = open(f"{PREDICTIONS_PATH}/{model}/{month}/{subreddit}.txt", 'w', encoding='utf-8', errors='ignore')

    for prediction in predictions:
        file_to_write.write(str(prediction) + '\n')

    file_to_write.close()


def write_weekly_avgs(model, subreddit):
    """
    A generalized function that writes the averages to a file as a TSV.

    Parameters: model name and subreddit name.
    Returns: None. Writes averages to a file.
    """
    # Open file to write to
    file_obj = open(f"{METRICS_PATH}/weeklyavgs/{model}/{subreddit}.tsv", "w", encoding="utf-8", errors="ignore")
    header = f"Month\tWeek\tAverage\n"
    file_obj.write(header)

    for month in MONTHS:
        
        averages = get_weekly_averages(model, month, subreddit)

        # Track weeks to write into TSV
        week = 0
        for average in averages:

            file_obj.write(f"{month}\t{week}\t{average}\n")
            week += 1

    return None

def write_weekly_totals(model, subreddit):
    """
    A generalized function that writes the averages to a file as a TSV.

    Parameters: model name and subreddit name.
    Returns: None. Writes averages to a file.
    """
    # Open file to write to
    file_obj = open(f"{METRICS_PATH}/weeklyposnegtotals/{model}/{subreddit}.tsv", "w", encoding="utf-8", errors="ignore")
    header = f"Month\tWeek\tPos\tNeg\n"
    file_obj.write(header)

    for month in MONTHS:
        
        totals = get_weekly_totals(model, month, subreddit)

        # Track weeks to write into TSV
        week = 0
        for pair in totals:

            file_obj.write(f"{month}\t{week}\t{pair[0]}\t{pair[1]}\n")
            week += 1

    return None

def write_monthly_totals(model, subreddit):
    """
    A generalized function that writes the averages to a file as a TSV.

    Parameters: model name and subreddit name.
    Returns: None. Writes averages to a file.
    """
    # Open file to write to
    file_obj = open(f"{METRICS_PATH}/monthlyposnegtotals/{model}/{subreddit}.tsv", "w", encoding="utf-8", errors="ignore")
    header = f"Month\tPos\tNeg\n"
    file_obj.write(header)

    for month in MONTHS:
        
        # Grab the weekly totals for each month
        totals = get_weekly_totals(model, month, subreddit)
        neg = sum([t[0] for t in totals])
        pos = sum([t[1] for t in totals])

        # Write totals
        file_obj.write(f"{month}\t{pos}\t{neg}\n")

    return None


def write_monthly_avgs(model, subreddit):
    """
    A generalized function that writes the averages to a file as a TSV.

    Parameters: model name and subreddit name.
    Returns: None. Writes averages to a file.
    """
    # Open file to write to
    file_obj = open(f"{METRICS_PATH}/monthlyavgs/{model}/{subreddit}.tsv", "w", encoding="utf-8", errors="ignore")
    header = f"Month\tAverage\n"
    file_obj.write(header)

    for month in MONTHS:
        
        average = sum(get_weekly_averages(model, month, subreddit))/4
        file_obj.write(f"{month}\t{average}\n")


    return None

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
            write_pred_func(get_pred_func(f"{DATA_PATH}/{month}/{subreddit}.txt"), model, month, subreddit)


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
    """
    Attains predictions from the LogReg BOW model.

    Parameters: Path to a file containing Reddit Data
    Returns: List of predictions
    """

    # Use global to avoid training model too much.
    global bow_model
    global tokens_list

    tfidf_matrix, doc_term_matrix, tokens_in_a_list = create_logreg_matrices(real_data_file, tokens_list)
    return [int(num) for num in bow_model.predict(doc_term_matrix)]
    
def get_tfidf_predictions(real_data_file):
    """
    Attains predictions from the LogReg TFIDF model.

    Parameters: Path to a file containing Reddit Data
    Returns: List of predictions
    """

    # Use global to avoid training model too much.
    global tfidf_model
    global tokens_list

    tfidf_matrix, doc_term_matrix, tokens_in_a_list = create_logreg_matrices(real_data_file, tokens_list)
    return [int(num) for num in bow_model.predict(doc_term_matrix)] 

def get_weekly_averages(model, month, subreddit, posts_per_week=20):
    """
    Gets the weekly averages for a specific model and returns them as a list.

    Parameters: String representation of the wanted model. 
    Returns: list of weekly averages
    """
    
    # Open file and create lists
    file_obj = open(f"{PREDICTIONS_PATH}/{model}/{month}/{subreddit}.txt", encoding="utf-8", errors="ignore")
    averages = list()
    predictions = [int(num) for num in file_obj.readlines()]
    
    # Create running sum
    running_sum = 0
    count = 1
    for num in predictions:
        # If we counted the full number of tweets for the week, 
        # average it.
        if count == posts_per_week:
            averages.append(running_sum/20)
            running_sum = 0
            count = 1
        
        running_sum += num
        count += 1

    return averages

def get_weekly_totals(model, month, subreddit, posts_per_week=20):
    """
    Gets the weekly pog/neg totals for a specific model and returns them as a list.

    Parameters: String representation of the wanted model. 
    Returns: list of weekly totals
    """
    
    # Open file and create lists
    file_obj = open(f"{PREDICTIONS_PATH}/{model}/{month}/{subreddit}.txt", encoding="utf-8", errors="ignore")
    totals = list()
    predictions = [int(num) for num in file_obj.readlines()]
    
    # Create running sum
    running_sum = 0
    count = 1
    for num in predictions:
        # If we counted the full number of tweets for the week, 
        # append 0 and 1 totals.

        if count == posts_per_week:
            totals.append((running_sum, 20-running_sum)) # (# of 0s, # of 1s) Binary so only need to track 0s
            count = 1
            running_sum = 0

        if num == 0:
            running_sum += 1

        count += 1

    return totals

if __name__ == "__main__":
    print("Executing Metrics.py ...")
    
    # Run Lexicon Predictions:
    # create_predictions(get_lexicon_predictions, write_predictions, "lexiconanalyzer")

    # Run RNN Predictions:
    # rnn_model = train_rnn_model()
    # create_predictions(get_rnn_predictions, write_predictions, "rnn")

    # Run LogReg Predictions
    # bow_model, tfidf_model, tokens_list = train_bow_and_tfidf()
    # create_predictions(get_bow_predictions, write_predictions, "logregbow")
    # create_predictions(get_tfidf_predictions, write_predictions, "logregtfidf")
    

    # Create Weekly Averages for all models
    for model in MODELS:
        for subreddit in SUBREDDITS:
            write_weekly_avgs(model, subreddit)

    # Create Monthly Averages for all models
    for model in MODELS:
        for subreddit in SUBREDDITS:
            write_monthly_avgs(model, subreddit)

    # Create Weekly Totals
    for model in MODELS:
        for subreddit in SUBREDDITS:
            write_weekly_totals(model, subreddit)

    # Create Monthly Totals
    for model in MODELS:
        for subreddit in SUBREDDITS:
            write_monthly_totals(model, subreddit)
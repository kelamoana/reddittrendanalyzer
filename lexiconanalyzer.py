# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

import helpers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from recurringNeuralNetworks import get_sentences_and_classes
from helpers import print_progress_bar

SID_OBJ = SentimentIntensityAnalyzer() 

# function to print sentiments 
# of the sentence. 
def vader_sentiment_score(sentence): 

    global SID_OBJ

    # calculate polarity scores which gives a sentiment dictionary, 
    # Contains pos, neg, neu, and compound scores.
    polarity_dict = SID_OBJ.polarity_scores(sentence) 
    
    # print("Raw sentiment dictionary : ", polarity_dict) 
    # print("polarity percentage of sentence ", polarity_dict['neg']*100, "% :: Negative") 
    # print("polarity percentage of sentence ", polarity_dict['pos']*100, "% :: Positive") 

    # print("Overall polarity percentage of sentence", end = " :: ") 

    # Calculate overall sentiment by compound score
    return 0 if polarity_dict['neg'] > polarity_dict['pos'] else 1

if __name__ == "__main__":
    sentences = helpers.get_sentences_and_classes()
    correct_guess = 0
    total_num = len(sentences)
    i = 0

    print(f"Running VADER on corpus...")
    for sentence, sentiment in sentences:

        if vader_sentiment_score(sentence) == int(sentiment):
            correct_guess += 1

        print_progress_bar(i, total_num)
        i += 1

    print(f"\n\nAccuracy Rate of VADER: {correct_guess/total_num}")
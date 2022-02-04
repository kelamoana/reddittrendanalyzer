# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

emotions = { i:w for i,w in 
    enumerate(
    ["admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment","disapproval", "disgust", 
    "embarrassment","excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral"])
    }


def preprocess_data_tsv(filename = "data/train.tsv"):
    """
    A function to process all of the training data and turn it
    into a usable data structure.

    Parameters: A TSV file
    Returns: A dict where each val is a tuple containing 
    the text and a list of its sentiments.
    """
    file_obj = open(filename, "r", encoding='utf-8', errors='ignore')

    file_lines = {}

    for i in range(10):

        line = file_obj.readline()
        line_to_append = line.replace("\n", '').split('\t')
        line_to_append[1] = [int(i) for i in line_to_append[1].replace(' ', '').split(',')]
        
        # FOR DEBUG
        # print("Text: ", line_to_append[0])
        # print("Emotions: ", end='')
        # for emotion in line_to_append[1]:   
        #     print(emotions[emotion], end=' ')
        # print() # Print a newline

        file_lines[line_to_append[2]] = (line_to_append[0], line_to_append[1])
    
    return file_lines

def convert_to_wordemb(prerpocessed_data):

    return None

if __name__ == "__main__":

    print(preprocess_data_tsv())
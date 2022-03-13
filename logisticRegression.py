from multiprocessing.sharedctypes import Value
import helpers
import numpy as np
import math
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from collections import Counter


# NOTE: LEAVE THE TEST.TSV FILE ALONE - THIS SHOULD BE ACTUAL TEST
# create train & validation data files in the format of BOW (text files)

def createBinaryClassificationFiles(filename):
    #modify this file to run it through test data instead of the training data
    file_obj = open(filename, "r", encoding='utf-8', errors='ignore')
    Xbinary = open('data/logisticRegression/XtrainData.txt', 'w')
    Ybinary = open('data/logisticRegression/YtrainData.txt', 'w')
    #get statistics on how much positive vs negative data we have
    numPositive = 0
    numNegative = 0
    for line in file_obj:
        lst_line = line.rstrip().split("\t")
        lst_categories = [int(num) for num in lst_line[1].split(",")]

        #mark as positive, negative, or don't include it(ambiguous/neutral)
        categoryClassification = 0
        for category in lst_categories:
            if (helpers.emotions[category] in helpers.emotionsCategories["ambiguous"]) or category == 27:
                categoryClassification += 0
            elif (helpers.emotions[category] in helpers.emotionsCategories["positive"]):
                categoryClassification += 1
            else:
                categoryClassification -= 1

        if categoryClassification != 0:
            if categoryClassification > 0 and numPositive < 9059:
                Xbinary.write(lst_line[0]+"\n")
                Ybinary.write(str(1)+"\n")
                numPositive += 1
            elif categoryClassification < 0:
                Xbinary.write(lst_line[0]+"\n")
                Ybinary.write(str(0)+"\n")
                numNegative += 1
            else:
                continue

    Ybinary.close()
    Xbinary.close()
    file_obj.close()
    print(numPositive)
    print(numNegative)

def splitData():
    XbinaryObj = open('data/logisticRegression/Xbinary.txt')
    YbinaryObj = open('data/logisticRegression/Ybinary.txt')
    XbinaryList = [line for line in XbinaryObj]
    YbinaryList = [line for line in YbinaryObj]

    Xtr, Xva, Ytr, Yva = train_test_split(XbinaryList,YbinaryList, train_size = 0.77, random_state = 5)

    xTrainFile = open('data/logisticRegression/XTrainData.txt', 'w')
    xValidationFile = open('data/logisticRegression/XValidationData.txt', 'w')
    yTrainFile = open('data/logisticRegression/YTrainData.txt', 'w')
    yValidationFile = open('data/logisticRegression/YValidationData.txt', 'w')

    Xtr = list(Xtr)
    Xva = list(Xva)
    Ytr = list(Ytr)
    Yva = list(Yva)

    for element in Xtr:
        xTrainFile.write(element)

    for element in Xva:
        xValidationFile.write(element)

    for element in Ytr:
        yTrainFile.write(element)

    for element in Yva:
        yValidationFile.write(element)

    xTrainFile.close()
    xValidationFile.close()
    yTrainFile.close()
    yValidationFile.close()

def tokenize(filename, matrix):
    setOfTokens = set()
    file_obj = open(filename, errors='ignore')
    curr_line_lst = []
    for line in file_obj:
        modify_line = line.rstrip("\n").lower()
        new_line = ""
        for char in modify_line:
            #removing punctuation
            if char not in [',', '.', ';', '?','!',':']:
                new_line += char
        new_line_lst = new_line.split()
        
        for word in new_line_lst:
            #ignore if it's a stopword
            if word not in stopwords.words('english'):
                curr_line_lst.append(word)
                setOfTokens.add(word)
        matrix.append(curr_line_lst)
        curr_line_lst = [] #reset for the next review
    #print("number of reviews:", len(matrix))
    file_obj.close()
    return list(setOfTokens)

def createBOWmodel(documentTermMatrix, tokensMatrix, tokensList):
    for document in tokensMatrix:
        documentTermRow = np.zeros(len(tokensList))
        for token in document:
            try:
                indexOfToken = tokensList.index(token)
                documentTermRow[indexOfToken] += 1
            except ValueError:
                continue
        documentTermMatrix.append(documentTermRow)
    return documentTermMatrix

def createTFIDFmodel(bow_model):

    bow_model = np.array(bow_model,dtype=object)

    N = len(bow_model)
    #calculate IDF for all words
    IDF = {}
    for i in range(len(bow_model[0])):
        IDF[i] = math.log(len(bow_model)/ (1+sum(bow_model[:,i])))

    #replace values in tokensMatrix with tfidf

    tfIdfModel = []

    for i in range(len(bow_model)):
        documentTermRow = np.zeros(len(bow_model[0]))
        #query for all elements where val > 0
        def condition(x): return x > 0
        output = [idx for idx, element in enumerate(bow_model[i]) if condition(element)]
        for num in output:
            documentTermRow[num] = (bow_model[i][num] / (1+ sum(bow_model[i])))*IDF[num]
        tfIdfModel.append(documentTermRow)


    return tfIdfModel


def reduceVocab(documentTermMatrix, tokensList):
    rareWordsIndices = set()
    npVersion = np.array(documentTermMatrix)
    for col_index in range(npVersion.shape[1]):
        colValues = npVersion[:,col_index]
        if sum(colValues) <= 1:
            rareWordsIndices.add(col_index)

    print('extracted rare word occurrances')
    reducedDocumentTermMatrix = []
    for row_index in range(len(npVersion)):
        newRow = []
        for col_index in range(len(npVersion[row_index])):
            if col_index not in rareWordsIndices:
                newRow.append(npVersion[row_index][col_index])
        reducedDocumentTermMatrix.append(newRow)

    reducedTokensList = []
    for token_index in range(len(tokensList)):
        if token_index not in rareWordsIndices:
            reducedTokensList.append(tokensList[token_index])

    return reducedDocumentTermMatrix, reducedTokensList


def runLogisticRegressionModel(documentTermMatrix, documentTermMatrixVa):
    y_tr = np.genfromtxt("data/logisticRegression/YTrainData.txt")
    y_va = np.genfromtxt("data/Twitter/YtestBinary.txt")
    # print(len(documentTermMatrix[0]))
    # print(len(documentTermMatrixVa[0]))
    model = SGDClassifier()
    model.fit(documentTermMatrix, y_tr)
    testAcc = model.score(documentTermMatrixVa, y_va)
    print("Score of accuracy on test data:", testAcc)


def create_logreg_matrices(training_data="data/logisticRegression/XTrainData.txt", train_tokens_list=None):
    """
    Generalized function to train tfidf. 
    Parameters: Training data file.
    Returns: TFIDF LR Model
    """

    # Create empty Matrices
    tokenize_matrix = []
    doc_term_matrix = []

    # Tokenize Words
    tokens_in_a_list = tokenize(training_data, tokenize_matrix)
    # print("number of tokens", len(tokens_in_a_list))
    # print("len of tokenize_matrix", len(tokenize_matrix))
    # print("width of tokenize_matrix", len(tokenize_matrix[0]))
    # Create doc term and TFIDF matrix to return
    if (train_tokens_list != None):
        tokens_in_a_list = train_tokens_list
    doc_term_matrix = createBOWmodel(doc_term_matrix, tokenize_matrix, tokens_in_a_list)
    tfidf_matrix = createTFIDFmodel(doc_term_matrix)
    # print("should be num of documents", len(doc_term_matrix))
    # print("should be number of terms", len(doc_term_matrix[0]))
    return tfidf_matrix, doc_term_matrix, tokens_in_a_list

def create_reduced_matrix(doc_term_matrix, tokens):
    """
    Reduces the amount of terms and tokens in the matrix and list, respectively.

    Parameters: Doc Term Matrix and List of Tokens.
    Returns: reduced DocTerm Matrix and Reduced Tokens.
    """

    # Reduced DocTerm Matrix and Reduced Tokens List
    return reduceVocab(doc_term_matrix, tokens)

def train_bow_and_tfidf(x_tr_data_file = "data/logisticRegression/XTrainData.txt", 
                        y_tr_data_file="data/logisticRegression/YTrainData.txt"):

    """
    Trains a BOW and TFIDF LogReg models and returns it.

    Parameters: Training Data File.
    Returns: Bow and TFIDF LogReg Models.
    """
    
    tfidf_matrix, doc_term_matrix, tokens_list = create_logreg_matrices(x_tr_data_file)
    reduced_doc_term_matrix, reduced_tokens_list = create_reduced_matrix(doc_term_matrix, tokens_list)
    # print("after reduce matrix function")
    # print("number of tokens", len(reduced_tokens_list))
    # print("should be num of documents", len(reduced_doc_term_matrix))
    # print("should be number of terms", len(reduced_doc_term_matrix[0]))

    y_tr = np.genfromtxt(y_tr_data_file)

    bow_model = SGDClassifier()
    bow_model.fit(reduced_doc_term_matrix, y_tr)

    tfidf_model = SGDClassifier()
    tfidf_model.fit(reduced_doc_term_matrix, y_tr)

    return bow_model, tfidf_model, reduced_tokens_list


if __name__ == "__main__":

    # Created Binary Classification Files
    # createBinaryClassificationFiles("data/test.tsv")
    # Split the Binary Classification Files into Training & Validation Files
    # splitData()

    # BOW Logistic Regression
    tokenizeMatrix = []
    docTermMatrix = []
    tokensInAList = tokenize("data/logisticRegression/XTrainData.txt", tokenizeMatrix)
    # print("number of reviews tokenizeMatrix:", len(tokenizeMatrix))
    docTermMatrix = createBOWmodel(docTermMatrix, tokenizeMatrix, tokensInAList)
    tfIdfMatrix = createTFIDFmodel(docTermMatrix)
    # print("num reviews docTermMatrix", len(docTermMatrix))
    # print(len(docTermMatrix))
    # print("num features in X", len(docTermMatrix[0]))

    reducedDocTermMatrix, reducedTokensLst = reduceVocab(docTermMatrix, tokensInAList)
    # print(len(reducedDocTermMatrix))
    # print(len(reducedDocTermMatrix[0]))
    # print(len(reducedTokensLst))
    
    # Create validation equivalent of BOW model
    tokenizeMatrixVa = []
    docTermMatrixVa = []

    # tokenize("data/logisticRegression/XValidationData.txt", tokenizeMatrixVa)
    tokenize("data/Twitter/XtestBinary.txt", tokenizeMatrixVa)
    docTermMatrixVa = createBOWmodel(docTermMatrixVa, tokenizeMatrixVa, reducedTokensLst)
    tfIdfTermMatrixVa = createTFIDFmodel(docTermMatrixVa)
    # print("num features in va X", len(docTermMatrixVa[0]))

    runLogisticRegressionModel(reducedDocTermMatrix, docTermMatrixVa)
    runLogisticRegressionModel(reducedDocTermMatrix, tfIdfTermMatrixVa)

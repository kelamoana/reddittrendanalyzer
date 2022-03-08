from multiprocessing.sharedctypes import Value
import helpers
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from collections import Counter
#NOTE: LEAVE THE TEST.TSV FILE ALONE - THIS SHOULD BE ACTUAL TEST 
# create train & validation data files in the format of BOW (text files)

def createBinaryClassificationFiles(filename):
    #modify this file to run it through test data instead of the training data
    file_obj = open(filename, "r", encoding='utf-8', errors='ignore')
    Xbinary = open('data/logisticRegression/XTrainData.txt', 'w')
    Ybinary = open('data/logisticRegression/YTrainData.txt', 'w')
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
    file_obj = open(filename)
    curr_line_lst = []
    for line in file_obj:
        modify_line = line.rstrip("\n").lower()
        new_line = ""
        for char in modify_line:
            #removing punctuation
            if char not in [',', '.', ';', '?','!',':']:
                new_line += char
        new_line_lst = new_line.split()
        #convert contraction -> expanded form
        for word in new_line_lst:
            #if "'" in word:
            #    contractionsDict.get(word)
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

def createTFIDFmodel(tokensMatrix, tokensList):

    N = len(tokensMatrix)
    
    #document frequency
    DF = {}
    for doc in tokensMatrix:
        for docToken in doc:
            try:
                DF[docToken]+=1
            except:
                DF[docToken] = 1
    print(len(DF) == len(tokensList))


    #tf idf
    tf_idf = {}
    for i in range(N):
        tokens = tokensMatrix[i]
        counter = Counter(tokens)
        for token in np.unique(tokens):
            tf = counter[token] / len(tokensList)
            df = DF[token]
            idf = np.log(N/(df+1))
            curr_tuple = tuple((doc,token))
            tf_idf[curr_tuple] = tf*idf
    

    #document vectorization
    D = np.zeros((N, len(tokensList)))
    for i in tf_idf:
        ind = tokensList.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    
    return D
    


        


def reduceVocab(documentTermMatrix, tokensList):
    rareWordsIndices = set()
    npVersion = np.array(documentTermMatrix)
    for col_index in range(npVersion.shape[1]):
        colValues = npVersion[:,col_index]
        if sum(colValues) <= 1:
            rareWordsIndices.add(col_index)
        '''if sum(colValues) == 2:
            rareWordsIndices.add(col_index)
        if sum(colValues) == 3:
            rareWordsIndices.add(col_index)
        if sum(colValues) == 4:
            countOfInstances += 1'''
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
    y_va = np.genfromtxt("data/logisticRegression/YTestData.txt")

    model = SGDClassifier()
    model.fit(documentTermMatrix, y_tr)
    testAcc = model.score(documentTermMatrixVa, y_va)
    print("Score of accuracy on test data:", testAcc)

# do the same for tf-idf
# then run on logistic regression
if __name__ == "__main__":
    """ Created Binary Classification Files """
    #createBinaryClassificationFiles("data/test.tsv")
    """ Split the Binary Classification Files into Training & Validation Files"""
    #splitData()
    """ BOW Logistic Regression """
    
    tokenizeMatrix = []
    docTermMatrix = []
    tokensInAList = tokenize("data/logisticRegression/XTrainData.txt", tokenizeMatrix)
    #print("number of reviews tokenizeMatrix:", len(tokenizeMatrix))
    docTermMatrix = createBOWmodel(docTermMatrix, tokenizeMatrix, tokensInAList)
    tfIdfMatrix = createTFIDFmodel(tokenizeMatrix, tokensInAList)
    #print("num reviews docTermMatrix", len(docTermMatrix))
    #print(len(docTermMatrix))
    print("num features in X",len(docTermMatrix[0]))

    reducedDocTermMatrix, reducedTokensLst = reduceVocab(docTermMatrix, tokensInAList)
    tfIdfReducedDocTermMatrix, tfIdfReducedTokensLst= reduceVocab(tfIdfMatrix,tokensInAList)
    print(len(reducedDocTermMatrix))
    print(len(reducedDocTermMatrix[0]))
    print(len(reducedTokensLst))
    #create validation equivalent of BOW model stuff
    tokenizeMatrixVa = []
    docTermMatrixVa = []
    tokenize("data/logisticRegression/XValidationData.txt", tokenizeMatrixVa)
    docTermMatrixVa = createBOWmodel(docTermMatrixVa, tokenizeMatrixVa, reducedTokensLst)
    tfIdfTermMatrixVa = createTFIDFmodel(tokenizeMatrixVa, reducedTokensLst)
    print("num features in va X", len(docTermMatrixVa[0]))

    runLogisticRegressionModel(reducedDocTermMatrix, docTermMatrixVa)
    runLogisticRegressionModel(tfIdfReducedDocTermMatrix,tfIdfTermMatrixVa)
    
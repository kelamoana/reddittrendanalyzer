import helpers
import numpy as np
from sklearn.model_selection import train_test_split
#NOTE: LEAVE THE TEST.TSV FILE ALONE - THIS SHOULD BE ACTUAL TEST 
# create train & validation data files in the format of BOW (text files)

def createBinaryClassificationFiles(filename):
    file_obj = open(filename, "r", encoding='utf-8', errors='ignore')
    Xbinary = open('data/logisticRegression/Xbinary.txt', 'w')
    Ybinary = open('data/logisticRegression/Ybinary.txt', 'w')

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
            Xbinary.write(lst_line[0]+"\n")
            if categoryClassification > 0:
                Ybinary.write(str(1)+"\n")
            else:
                Ybinary.write(str(0)+"\n")

    Ybinary.close()
    Xbinary.close()
    file_obj.close()   

def splitData():
    #X = np.genfromtxt('data/logisticRegression/Xbinary.txt')
    #Y = np.genfromtxt('data/logisticRegression/Ybinary.txt')
    XbinaryObj = open('data/logisticRegression/Xbinary.txt')
    YbinaryObj = open('data/logisticRegression/Ybinary.txt')
    XbinaryList = [line for line in XbinaryObj]
    YbinaryList = [line for line in YbinaryObj]
    #X = np.genfromtxt(, dtype=str)
    #Y = np.loadtxt('data/logisticRegression/Ybinary.txt')
    
    Xtr, Xva, Ytr, Yva = train_test_split(XbinaryList,YbinaryList, train_size = 0.7, random_state = 5)
    
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

# do the same for tf-idf
# then run on logistic regression
if __name__ == "__main__":
    """ Created Binary Classification Files """
    #createBinaryClassificationFiles("data/train.tsv")
    """ Split the Binary Classification Files into Training & Validation Files"""
    #splitData()
    


# File: bayes_CV.py
# Author(s) names AND netid's: Haikun Liu    hlg483
# Date: 05.19.2016
# Statement: I worked individually on this project and all work is my own.

import math, os, pickle, re, random, nltk

def crossValidation(fold, trial):
    """This method conducts cross validate for bayes, the functions used in this method are almost
        the same as bayes, but with some changes to be suitable for performance measure
        The reason why to separate crossValidation function from bayes class is using the member functions in
        bayes will overwrite the bayes model we need

        input: # of fold, # of trial
        output: accuracy, precision, recall, F1"""

    # load file list
    lFileList = []
    for fFileObj in os.walk("./movies_reviews/"):
        lFileList = fFileObj[2]
        break
    # filtering file list
    fileList = [file for file in lFileList if
                re.search("movies-1", file) or re.search("movies-5", file)]
    # using confusionMat_trial to store confusion matrices of each trial
    confusionMat_trial = {}
    for j in range(trial):
        # rearrange file list
        random.shuffle(fileList)
        # calculate # of files for a fold
        subSetSize = len(fileList) / fold
        # using confusionMat_fold to store confusion matrices of each fold
        confusionMat_fold = {}
        for i in range(fold):
            # separate test set and train set
            testSet = fileList[i * subSetSize:][:subSetSize]
            trainSet = fileList[:i * subSetSize] + fileList[(i + 1) * subSetSize:]

            ########### training #############
            model = train(trainSet)

            ########### classification ########

            # initialize confusion matrix
            confusionMat = {"truePositive": 0, "falseNegative": 0, "trueNegative": 0, "falsePositive": 0}

            # testing
            for file in testSet:
                review = loadFile("./movies_reviews/" + str(file))
                classLabel = "positive" if re.search("movies-5", file) else "negative"

                label = classify(model, review)

                if classLabel == "positive":
                    if label == "positive":
                        confusionMat["truePositive"] += 1
                    else:
                        confusionMat["falseNegative"] += 1
                else:
                    if label == "negative":
                        confusionMat["trueNegative"] += 1
                    else:
                        confusionMat["falsePositive"] += 1
        # store confusion matrix
            confusionMat_fold[i] = confusionMat
        confusionMat_trial[j] = confusionMat_fold

    ########### Evaluation #########
    sumAccuracy = 0.0
    sumPrecision_pos = 0.0
    sumRecall_pos = 0.0
    sumF1_pos = 0.0
    sumPrecision_neg = 0.0
    sumRecall_neg = 0.0
    sumF1_neg = 0.0
    total = 0

    for _, trailValue in confusionMat_trial.items():
        for _, foldValue in trailValue.items():
            total += 1
            try:
                accuracy = float(foldValue["truePositive"] + foldValue["trueNegative"]) / float(sum(foldValue.values()))
            except ZeroDivisionError:
                accuracy = 0.0
            try:
                precision_pos = float(foldValue["truePositive"]) / float(foldValue["truePositive"] + foldValue["falsePositive"])
                precision_neg = float(foldValue["trueNegative"]) / float(foldValue["trueNegative"] + foldValue["falseNegative"])
            except ZeroDivisionError:
                precision_pos = 0.0
                precision_neg = 0.0
            try:
                recall_pos = float(foldValue["truePositive"]) / float(foldValue["truePositive"] + foldValue["falseNegative"])
                recall_neg = float(foldValue["trueNegative"]) / float(foldValue["trueNegative"] + foldValue["falsePositive"])
            except ZeroDivisionError:
                recall_pos = 0.0
                recall_neg = 0.0
            try:
                F1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
                F1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)
            except ZeroDivisionError:
                F1_pos = 0.0
                F1_neg = 0.0
            sumAccuracy += accuracy
            sumPrecision_pos += precision_pos
            sumRecall_pos += recall_pos
            sumF1_pos += F1_pos
            sumPrecision_neg += precision_neg
            sumRecall_neg += recall_neg
            sumF1_neg += F1_neg
            # print matrices for each measurement
            print "Positive = ", accuracy, precision_pos, recall_pos, F1_pos
            print "Negative = ", accuracy, precision_neg, recall_neg, F1_neg
    # print overall result
    print "Positive: ","accuracy =", sumAccuracy / total, ", precision =", sumPrecision_pos / total, ", recall =", sumRecall_pos / total, ", F1 =", sumF1_pos / total
    print "Negative: ", "accuracy =", sumAccuracy / total, ", precision =", sumPrecision_neg / total, ", recall =", sumRecall_neg / total, ", F1 =", sumF1_neg / total


def train(trainSet):
    """Trains the Naive Bayes Sentiment Classifier."""
    model = {}
    posfileList = [file for file in trainSet if re.search("movies-5", file)]
    negfileList = [file for file in trainSet if re.search("movies-1", file)]

    model["posDict"] = constructDict("./movies_reviews/", posfileList)
    model["negDict"] = constructDict("./movies_reviews/", negfileList)
    model["reviewCounter"] = {"negative": len(negfileList), "positive": len(posfileList)}
    return model


def constructDict(directory, fileList):
    wordDict = {}
    for file in fileList:
        review = loadFile(directory + str(file))
        wordList = tokenize(review)
        for word in wordList:
            if word in wordDict.keys():
                wordDict[word] += 1
            else:
                wordDict[word] = 1
    return wordDict


def classify(model, sText):
    """Given a target string sText, this function returns the most likely document
    class to which the target string belongs (i.e., positive, negative or neutral).
    """
    posDict = model["posDict"]
    negDict = model["negDict"]
    reviewCounter = model["reviewCounter"]

    posPrior = float(reviewCounter["positive"]) / float(sum(reviewCounter.values()))
    negPrior = float(reviewCounter["negative"]) / float(sum(reviewCounter.values()))
    posScore = math.log(posPrior)
    negScore = math.log(negPrior)

    wordList = tokenize(sText)

    for word in wordList:
        if word in posDict.keys():
            posScore += math.log(
                float(posDict[word] + 1) / float(sum([value + 1 for _, value in posDict.items()])))
        else:
            posScore += math.log(float(1) / float(sum([value + 1 for _, value in posDict.items()])))
        if word in negDict.keys():
            negScore += math.log(
                float(negDict[word] + 1) / float(sum([value + 1 for _, value in negDict.items()])))
        else:
            negScore += math.log(float(1) / float(sum([value + 1 for _, value in negDict.items()])))

    label = "positive" if posScore > negScore else "negative"
    return label


def loadFile(sFilename):
    """Given a file name, return the contents of the file as a string."""

    f = open(sFilename, "r")
    sTxt = f.read()
    f.close()
    return sTxt


def save(dObj, sFilename):
    """Given an object and a file name, write the object to the file using pickle."""

    f = open(sFilename, "w")
    p = pickle.Pickler(f)
    p.dump(dObj)
    f.close()


def load(sFilename):
    """Given a file name, load and return the object stored in the file."""

    f = open(sFilename, "r")
    u = pickle.Unpickler(f)
    dObj = u.load()
    f.close()
    return dObj


def tokenize(sText):
    """Given a string of text sText, returns a list of the individual tokens that
    occur in that string (in order)."""

    lTokens = []
    sToken = ""
    for c in sText:
        if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
        else:
            if sToken != "":
                lTokens.append(sToken)
                sToken = ""
            if c.strip() != "":
                lTokens.append(str(c.strip()))

    if sToken != "":
        lTokens.append(sToken)

    return lTokens

if __name__ == "__main__":
    crossValidation(10, 1)






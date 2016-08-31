# File: bayesbest.py
# Author(s) names AND netid's: Haikun Liu    hlg483
# Date: 05.19.2016
# Statement: I worked individually on this project and all work is my own.

import math, os, pickle, re, random
import string, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Bayes_Classifier:
    def __init__(self):
        """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text."""

        # load naive bayes model. if IOError, then train the model
        try:
            self.model = self.load("bayesbest.bat")
        except IOError:
            self.model = {}
            self.train()

    def train(self):
        """Trains the Naive Bayes Sentiment Classifier."""

        # load file list
        lFileList = []
        for fFileObj in os.walk("./movies_reviews/"):
            lFileList = fFileObj[2]
            break

        # classify files into positive files and negative files
        posfileList = [file for file in lFileList if re.search("movies-5", file)]
        negfileList = [file for file in lFileList if re.search("movies-1", file)]

        # construct naive bayes model, inculding positive dictionary, negative dictionary,
        # and # of positive review + # of negative review
        self.model["posDict"] = self.constructDict("./movies_reviews/", posfileList)
        self.model["negDict"] = self.constructDict("./movies_reviews/", negfileList)
        self.model["reviewCounter"] = {"negative": len(negfileList), "positive": len(posfileList)}
        self.save(self.model, "bayesbest.bat")

    def constructDict(self, directory, fileList):
        """Help function for construct word dictionary
            input: file directory, list of files
            output: word dictionary"""
        wordDict = {}
        for file in fileList:
            review = self.loadFile(directory + str(file))
            wordList = self.tokenize(review)
            for word in wordList:
                if word in wordDict.keys():
                    wordDict[word] += 1
                else:
                    wordDict[word] = 1
        return wordDict


    def classify(self, sText):
        """Given a target string sText, this function returns the most likely document
        class to which the target string belongs (i.e., positive, negative or neutral).
        """
        # load naive bayes model, including positive dictionary, negative dictionary
        # reviewCounter contains # of positive reviews and # of negative reviews
        posDict = self.model["posDict"]
        negDict = self.model["negDict"]
        reviewCounter = self.model["reviewCounter"]

        # epsilon is set to label "neutral"
        epsilon = 0.5

        # calculate priors
        posPrior = float(reviewCounter["positive"]) / float(sum(reviewCounter.values()))
        negPrior = float(reviewCounter["negative"]) / float(sum(reviewCounter.values()))

        # initialize positive score and negative score
        posScore = math.log(posPrior)
        negScore = math.log(negPrior)

        # tokenize input string
        wordList = self.tokenize(sText)

        # calculate positive score and negative score
        for word in wordList:
            if word in posDict.keys():
                posScore += math.log(float(posDict[word] + 1) / float(sum([value + 1 for _, value in posDict.items()])))
            else:
                posScore += math.log(float(1) / float(sum([value + 1 for _, value in posDict.items()])))
            if word in negDict.keys():
                negScore += math.log(float(negDict[word] + 1) / float(sum([value + 1 for _, value in negDict.items()])))
            else:
                negScore += math.log(float(1) / float(sum([value + 1 for _, value in negDict.items()])))

        # labeling
        if posScore > negScore and posScore - negScore > epsilon:
            return "positive"
        elif negScore > posScore and negScore - posScore > epsilon:
            return "negative"
        else:
            return "neutral"


    def loadFile(self, sFilename):
        """Given a file name, return the contents of the file as a string."""

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt


    def save(self, dObj, sFilename):
        """Given an object and a file name, write the object to the file using pickle."""

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()


    def load(self, sFilename):
        """Given a file name, load and return the object stored in the file."""

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        """Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order)."""

        # convert the string into lower-case
        sText = sText.lower()
        # remove the punctuation using the character deletion step of translate
        sText = sText.translate(None, string.punctuation)
        # use nltk tokenizer to tokenize the string into word list
        lTokens = nltk.word_tokenize(sText)
        # convert the word in lTokens to unicode, because nltk package uses unicode
        lTokens = [unicode(w, errors= 'ignore') for w in lTokens]
        # use nltk.corpus.stopwords list to eliminate neutral words in lTokens
        lTokens = [w for w in lTokens if not w in stopwords.words('english')]
        # conduct stemming using nltk.stem.porter.PorterStemmer
        lTokens = [PorterStemmer().stem(w) for w in lTokens]
        # convert lTokens from unicode back to utf-8 for easy reading
        lTokens = [w.encode('utf-8') for w in lTokens]
        return lTokens


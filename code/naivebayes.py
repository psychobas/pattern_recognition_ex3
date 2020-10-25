import numpy as np
import math
import glob
import re
from typing import List


class Word():
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''
    def __init__(self, word, numOfHamWords, numOfSpamWords, indicativeness):
        self.word = word
        self.numOfHamWords = numOfHamWords
        self.numOfSpamWords = numOfSpamWords
        self.indicativeness = indicativeness


class NaiveBayes():
    '''
    Naive bayes class
    Train model and classify new emails
    '''
    def _extractWords(self, filecontent: str) -> List[str]:
        '''
        Word extractor from filecontent
        :param filecontent: filecontent as a string
        :return: list of words found in the file
        '''
        txt = filecontent.split(" ")
        txtClean = [(re.sub(r'[^a-zA-Z]+', '', i).lower()) for i in txt]
        words = [i for i in txtClean if i.isalpha()]
        print(words)
        return words

    def train(self, msgDirectory: str, fileFormat: str = '*.txt') -> (List[Word], float):
        '''
        :param msgDirectory: Directory to email files that should be used to train the model
        :return: model dictionary and model prior
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))

        ham_words = []
        spam_words = []

        spam_words_flat = []
        ham_words_flat = []

        for file in files:
            if "spmsga" in file:
                f_open = open(file)
                f = f_open.read()
                f = self._extractWords(f)
                spam_words.append(f)
                # is still a list of lists.
                #flatten
                for sublist in spam_words:
                    for item in sublist:
                        spam_words_flat.append(item)



                print("spam")
            else:
                print("ham")
                f_open = open(file)
                f = f_open.read()
                f = self._extractWords(f)
                ham_words.append(f)

                # flatten
                for sublist in ham_words:
                    for item in sublist:
                        ham_words_flat.append(item)

        #dictionary with word counts for spam words:
        spam_word_count_dict = {}
        ham_word_count_dict = {}


        for word in spam_words_flat:
            try:
                spam_word_count_dict[word] += 1
            except KeyError:
                spam_word_count_dict[word] = 1


        for word in ham_words_flat:
            try:
                ham_word_count_dict[word] += 1
            except KeyError:
                ham_word_count_dict[word] = 1

        # todo: loop over both dictionaries to create list with Word instances


        unique_words = set(spam_words_flat + ham_words_flat)

        final_dictionary = []  #

        #total number of words per class
        n_words_spam = sum(spam_word_count_dict.values())
        n_words_ham = sum(ham_word_count_dict.values())

        for word in unique_words:

            numOfHamWords = ham_word_count_dict.get(word, 0) + 1 #smoothing
            numOfSpamWords = spam_word_count_dict.get(word, 0) + 1 #smoothing

            #double check if smoothing (denominator) is correct
            spam_likelihood = numOfSpamWords / n_words_spam + len(spam_word_count_dict)
            ham_likelihood = numOfHamWords / n_words_ham + len(ham_word_count_dict)

            indicativeness = spam_likelihood / ham_likelihood

            w = Word(word = word, numOfHamWords = numOfHamWords, numOfSpamWords = numOfSpamWords, indicativeness = indicativeness)

            final_dictionary.append(w)





        # TODO: Train the naive bayes classifier
        # TODO: Hint - store the dictionary as a list of 'wordCounter' objects

        #use _extractWords function to extract the words once they are loaded

        spam_emails_count = 0

        priorSpam = len(spam_words) / len(ham_words) # fraction of documents that are spam (start with "spm")
        self.logPrior = math.log(priorSpam / (1.0 - priorSpam))
        final_dictionary.sort(key=lambda x: x.indicativeness, reverse=True)
        self.dictionary = final_dictionary
        return self.dictionary, self.logPrior

    def classify(self, message: str, number_of_features: int) -> bool:
        '''
        :param message: Input email message as a string
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''

        txt = np.array(self._extractWords(message))
        # TODO: Implement classification function

        #USE LAPLACE SMOOTHING FOR LIKELIHOOD (SLIDE 9)

        return 0#???

    def classifyAndEvaluateAllInFolder(self, msgDirectory: str, number_of_features: int,
                                       fileFormat: str = '*.txt') -> float:
        '''
        :param msgDirectory: Directory to email files that should be classified
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: Classification accuracy
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        corr = 0  # Number of correctly classified messages
        ncorr = 0  # Number of falsely classified messages
        # TODO: Classify each email found in the given directory and figure out if they are correctly or falsely classified
        # TODO: Hint - look at the filenames to figure out the ground truth label
        return corr / (corr + ncorr)

    def printMostPopularSpamWords(self, num: int) -> None:
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary

    def printMostPopularHamWords(self, num: int) -> None:
        print("{} most popular HAM words:".format(num))
        # TODO: print the 'num' most used HAM words from the dictionary

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary

    def printMostindicativeHamWords(self, num: int) -> None:
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


vectorizer = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
        dtype='numpy.int64', encoding='utf-8', input='content',
        lowercase=True, 
        max_df=1.0, 
        max_features=None, min_df=1,
        ngram_range=(2, 2), 
        preprocessor=None, 
        stop_words=None,
        strip_accents=None, 
        token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)


corpus = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
     'pen pen de chorvalu, de kemerlu, de eklavu'
     'kumakarisma si'
]

vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=20)



								   
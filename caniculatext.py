from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
from set_up_data import get_data as get_data
from pprint import pprint as pprint
from sklearn import svm

# vectorizer = CountVectorizer(analyzer='word',
#                              binary=False,
#                              decode_error='strict',
#                              dtype='numpy.int64',
#                              encoding='utf-8',
#                              input='content',
#                              lowercase=True,
#                              max_df=1.0,
#                              max_features=None, min_df=1,
#                              ngram_range=(2, 2),
#                              preprocessor=None,
#                              stop_words=None,
#                              strip_accents=None,
#                              token_pattern='(?u)\\b\\w\\w+\\b',
#                              tokenizer=None,
#                              vocabulary=None)


# corpus = [
#      'This is the first document.',
#      'This is the second second document.',
#      'And the third one.',
#      'Is this the first document?',
#      'pen pen de chorvalu, de kemerlu, de eklavu'
#      'kumakarisma si'
# ]

# vectorizer = HashingVectorizer(stop_words='english',
#                                non_negative=True,
#                                n_features=20)
# print()

print('pogi ako')

training_data = get_data()

count_vect = CountVectorizer()
vectorizer = TfidfVectorizer(sublinear_tf=True,
                             max_df=0.5,
                             stop_words='english')
tfidf_transformer = TfidfTransformer()
clf = svm.SVC(decision_function_shape='ovo')

X_train_counts = count_vect.transform(training_data['names'])
								   
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
xCoordinateTrainingData = vectorizer.transform(training_data['names'])
test = clf.fit(xCoordinateTrainingData, training_data['classes'])

tasks_new = ['add', 'delete', 'show']

X_new_counts = count_vect.transform(tasks_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
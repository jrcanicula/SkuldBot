"""
 Author: Jaime R. Canicula
         Pia Carmela M. Quizon

 Project name: TaskBot


--------------------------------------------------------
 Some sort of 'Description that can help the program'
--------------------------------------------------------

 Consider n-features as unique number of words in our dataset (or from the database); using scipy.sparse
 
 Consider classification_number as number of stratas defined for each datasets; 
 
 Definition : vectorization as the general process of turning a collection of text documents into numerical feature vectors. 
 
 The strategy of this python script lies in this process 
 
 tokenization -> counting -> normalization

 Bag of Words or 'Bag of n-grams' representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.


As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have many feature values that are zeros (typically more than 99 percent them).
For instance a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.
In order to be able to store such a matrix in memory but also to speed up algebraic operations matrix / vector, implementations will typically use a sparse representation such as the implementations available in the scipy.sparse package.

"""

from __future__ import print_function

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


X = [[0], [1], [2], [3]]

Y = [0, 1, 2, 3]

clf = svm.SVC(decision_function_shape='ovo')

clf.fit(X, Y) 

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
dec = clf.decision_function([[1]])

print(dec.shape[1])# 4 classes: 4*3/2 = 6
#
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
print(dec.shape[1]) # 4 classes
#




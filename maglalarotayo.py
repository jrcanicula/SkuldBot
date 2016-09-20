
from set_up_data import get_data as get_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


from pprint import pprint as pprint

training_data = get_data()

#print (training_data)

#print(training_data['names'][0])

count_vect = CountVectorizer()

x_train_counts = count_vect.fit_transform(training_data['names'])

#print(x_train_counts.shape)

# count_vect.vocabulary_.get(u'1')

# print words print (count_vect.vocabulary_.get(u'delete'))

# print(x_train_counts[])


tfidf_transformer = TfidfTransformer()

x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

#print(x_train_tfidf.shape)

#clf = svm.LinearSVC().fit(x_train_tfidf, training_data['classes'])

clf =svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=3.2, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=42, max_iter=1000).fit(x_train_tfidf,training_data['classes'])

#C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  #  max_iter=-1, probability=False, random_state=None, shrinking=True,
   # tol=0.001, verbose=False

tasks_new = ['Get on, saranghae, delete, deletion haircut, mitsukake, delete all tasks saranghae delete insert add add today new task Get haircut on Saturday','kinakarir ng lahat','simple','saranghae aishiteru, konbanwa','salty','maritime,kiligami','taramisoe123','saranghae','Selemene','list all the documents i have','i can list all you want','misclassified shits within the mic']

x_new_counts = count_vect.transform(tasks_new)

x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = clf.predict(x_new_tfidf)

#print(metrics.accuracy_score(predicted,predicted))

print (predicted);
print(x_train_tfidf.shape)
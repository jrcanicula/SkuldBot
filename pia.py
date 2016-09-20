
from set_up_data import get_data as get_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm



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

clf = MultinomialNB().fit(x_train_tfidf, training_data['classes'])

tasks_new = ['add pia to the users where users are not something', 'kemerot parrot dumeletesung ang delete', 'show all tasks', 'add new tasks to the user table','halberds','delete someone to love','making sense men show']

x_new_counts = count_vect.transform(tasks_new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = clf.predict(x_new_tfidf)

pprint(predicted)

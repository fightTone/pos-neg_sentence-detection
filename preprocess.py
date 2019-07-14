import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
	def classify(self,features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	def confidence(self,features):
		for c in self._classifiers:
			votes = []
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf

short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_documents = open("documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]
save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev,category) in documents]
save_classifier = open("featuresets.pickle", "wb")
pickle.dump(featuresets, save_classifier)
save_classifier.close()
random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# training_set = featuresets[100:]
# testing_set = featuresets[:100]


classifier = nltk.NaiveBayesClassifier.train(training_set)

# classifier_f = open("naivebayes.pickle","rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()

print ("Original Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MNB_classifier Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
save_classifier = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print ("GaussianNB_classifier  accuracy: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)
# save_classifier = open("GaussianNB_classifier.pickle", "wb")
# pickle.dump(GaussianNB_classifier, save_classifier)
# save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB_classifier accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
save_classifier = open("BernoulliNB_classifier.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("LogisticRegression_classifier  accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
save_classifier = open("LogisticRegression_classifier.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print ("SGDClassifier_classifier accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
# save_classifier = open("SGDClassifier_classifier.pickle", "wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print ("SVC_classifier accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
# save_classifier = open("SVC_classifier.pickle", "wb")
# pickle.dump(SVC_classifier, save_classifier)
# save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("LinearSVC_classifier accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
save_classifier = open("LinearSVC_classifier.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print ("NuSVC_classifier  accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
# save_classifier = open("NuSVC_classifier.pickle", "wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()

voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,LinearSVC_classifier)
print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
save_classifier = open("voted_classifier.pickle", "wb")
pickle.dump(voted_classifier, save_classifier)
save_classifier.close()

# print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification: ", voted_classifier.classify(testing_set[5][0]), "Confidence %:", voted_classifier.confidence(testing_set[5][0])*100)
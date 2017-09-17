'''from flask import Flask, request
from twilio import twiml

app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms():
	number = request.form['From']
	message_body = request.form['Body']

	resp = twiml.Response()
	resp.message('Hello {}, you said: {}'.format(number, message_body))
	return str(resp)
	

if __name__ == '__main__':
	app.run(debug=True)'''

from flask import Flask, request, redirect
from twilio.twiml.messaging_response import MessagingResponse
from geopy.geocoders import Nominatim

geolocator = Nominatim()


''' BEGIN ML '''

import string
def read_corpus(fname):
    for line in open(fname):
        tokens = line.split()
        words = tokens[1:]
        yield ' '.join(words)
def read_labels(fname):
    to_return = []
    for line in open(fname):
        tokens = line.split()
        label = 1 if tokens[0] == 'ham' else 0
        to_return.append(label)
    return to_return
from scipy import spatial
import numpy

from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def load_dict_and_embedding():
    word_dict = dict()
    with open("word_dict", "r") as f:
        for line in f:
            key, value = line.strip().split(" ")
            word_dict[key] = int(value)

    embeddings = numpy.loadtxt("embedding_table", delimiter=",")
    return word_dict, embeddings

# load word dict and embedding table
word_dict, embedding_table = load_dict_and_embedding()
all_docs = read_corpus("SMSSpamCollection")
all_regressors = []
my_embedding = [0]*32
for doc in all_docs:
    #doc = strip_punctuation(doc)
    i = 0
    for word in doc.split(' '):
        if (word.lower() in word_dict):
            my_embedding += embedding_table[word_dict[word.lower()]]
            i+=1
    if i == 0:
        i = 1
    all_regressors.append([x/i for x in my_embedding])
from sklearn import linear_model
classifier = linear_model.LogisticRegression()
train_labels = read_labels('SMSSpamCollectionTrain')
test_labels = read_labels('SMSSpamCollectionTest')
classifier.fit(all_regressors[:750], train_labels)
test_predictions = classifier.predict(all_regressors[750:])
len_predictions = len(test_predictions)
corrects = sum(test_predictions == test_labels)
errors = len_predictions - corrects
error_rate = float(errors)/len_predictions
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300, max_df=0.1, min_df=0.01, sublinear_tf=True)
all_docs = read_corpus('SMSSpamCollection')
tfidf_matrix = vectorizer.fit_transform(all_docs)

import pandas as pd
feature_names = vectorizer.get_feature_names()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df

classifier2 = linear_model.LogisticRegression()
all_regressors2 = tfidf_matrix.todense().tolist()
train_labels = read_labels('SMSSpamCollectionTrain')
test_labels = read_labels('SMSSpamCollectionTest')
classifier2.fit(all_regressors2[:750], train_labels)
test_predictions = classifier2.predict(all_regressors2[750:])
len_predictions = len(test_predictions)
corrects = sum(test_predictions == test_labels)
errors = len_predictions - corrects
error_rate = float(errors)/len_predictions

all_docs = read_corpus("SMSSpamCollection")
all_docs = [doc for doc in all_docs]
classifier2 = linear_model.LogisticRegression()
classifier2.fit(all_regressors2[:750], train_labels)
import numpy as np
def paddle_predict(inpstring):
    my_embedding = [0]*32
    for word in inpstring.split(' '):
        if word.lower() in word_dict:
            my_embedding+=embedding_table[word_dict[word.lower()]]
    test_prediction = classifier.predict(my_embedding)
    all_docs.append(inpstring)
    train_labels = read_labels('SMSSpamCollectionTrain')
    all_regressors2 = vectorizer.fit_transform(all_docs).todense().tolist()
    test_prediction2 = classifier2.predict(all_regressors2[-1])
    return np.rint(0.3*test_prediction+0.7*test_prediction2) # 1 if ham
# paddle_predict("i love you babe, call 555 for a good time(for free free free free free free call me call me call me 38382737237 433842734823 phone free call me call me you won won won won  free)")

''' END ML '''


app = Flask(__name__)

@app.route("/sms", methods=['GET', 'POST'])
def sms_reply():
	message_body = request.form['Body']

	# PADDLEPADDLE

	if 'FromCity' in request.form and 'ToCity' in request.form:
		loc1 = geolocator.geocode(request.form['FromCity'])
		loc2 = geolocator.geocode(request.form['ToCity'])
		dist = (loc1.latitude - loc2.latitude)**2 + (loc1.longitude - loc2.longitude)**2
		print(dist)

		if dist > 100:
			return ''
	
	if np.rint(paddle_predict(message_body)[0]) == 0: # spam
		return ''

	print(message_body)
	f_out = open('out.txt', 'a')
	f_out.write('\n' + message_body)
	f_out.close()

	return ''

if __name__ == "__main__":
	app.run(debug=True)

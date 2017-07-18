# -*- coding: utf-8 -*-
"""
@author: quzhou
-- EDA for data
-- Convert a corpus to a vector format: bag-of-words approach
Massage the raw message (sequence of characters) into vectors ( sequences of numbers)
1. split a message into its individual words and return a list
* remove punctuation
* remove very common words('the','a',etc)
2. vectorization
* convert the each message(represented as a list of tokens) into a vector
* count DF in the vector
* weight the counts(IDF)
* Normalize the vectors to unit length( L2 norm)


"""
import string
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords


os.getcwd()
#'C:\\Users\\qz\\OneDrive\\Documents\\NLP'
os.chdir('C:\\Users\\qz\\OneDrive\\Documents\\NLP')

# (1) Data EDA
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print len(messages)

for message_no, message in enumerate(messages[:10]):
    print message_no, message
    print '\n'

df = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t', names=["label","message"])
df.head()
df.describe()

df.groupby('label').describe()

df['length']=df['message'].apply(len)
df['length'].plot(bins=100,kind='hist')
df.length.describe()
df.hist('length',by='label',bins=50)# spam tends to have more characters
sns.factorplot('length',data=df, hue='label',kind='count')



#remove punctuations
mess = 'Sample message! Notice: it has punctuation.'
nopunc = [char for char in mess if char not in string.punctuation] # characters(single letter) without punctuation
nopunc = ''.join(nopunc) # join the letters to one string

#remove stopwords
stopwords.words('english')[0:15]

nopunc.split()
#remove stop words
nopunc_clean=[word for word in nopunc.split() if word.lower not in stopwords.words('english') ]
nopunc_clean

# define a function to remove punctuation and stopwords
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower not in stopwords.words('english')]

#apply the messages into cleaned words : a test on 5 rows
df['message'].head().apply(text_process)


# Vectorization
#CountVectorizer convert documents to a matrix of token counts . This matrix would be a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(df['message'])

#print totoal_number of vocab words
print len(bow_transformer.vocabulary_)

# Let's take a look at one text message and get the bag-of-words counts as a vector, and put to use of the bow_transformer
message10 = df['message'][9]
print message10

bow10 = bow_transformer.transform([message10])
print bow10 # 20 words counts are non-zero
print bow10.shape # vector is 11747 tows

print bow_transformer.get_feature_names()[5810] # check the word at 5810 row
print bow_transformer.get_feature_names()[8471] # the two words are :color and mobile

# use .transform on our bag-of-words (bow) and transform entire dataframe to a bag-of-words count matrix
message_bow = bow_transformer.transform(df['message'])
print 'shape of sparse matrix:', message_bow.shape
print 'amount of Non-zero counts of words:', message_bow.nnz
print 'sparsity: %.2f%%' %(100.0* message_bow.nnz /(message_bow.shape[0] * message_bow.shape[1]))


#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


tfidf_transformer = TfidfTransformer().fit(message_bow)
tfidf10 = tfidf_transformer.transform(bow10)#transform the tf-idf of message10 bag-of-words
print tfidf10
#check what is the idf of the word 'good' and 'love'
print tfidf_transformer.idf_[bow_transformer.vocabulary_['good']]
print tfidf_transformer.idf_[bow_transformer.vocabulary_['love']]

#transform the entire bag-of-words corpus into TF-IDF matrix at once:
message_tfidf =tfidf_transformer.transform(message_bow)
print message_tfidf.shape  

#training a model using naive bayesian
spam_detect_model = MultinomialNB().fit(message_tfidf, df['label'])
print 'predicted:', spam_detect_model.predict(tfidf10)[0]
print 'expected:', df.label[9]
#evaluation
all_predictions = spam_detect_model.predict(message_tfidf)
print classification_report(df['label'], all_predictions)
print accuracy_score(all_predictions, df.label) # 97% accuracy



##training a model using gradient boosting tree
#from sklearn.ensemble import GradientBoostingClassifier  
#
#spam_gbc = GradientBoostingClassifier().fit(message_tfidf, df['label'])
#
#print 'prdicted:', spam_gbc.predict(tfidf10)[0]
#print 'labeled:',df.label[9]
#
#
#evaluation of model
from sklearn.cross_validation import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(df['message'], df['label'], test_size=0.3)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)


### data pipeline 
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    
  
    
pipeline.fit(msg_train,label_train)   
pred = pipeline.predict(msg_test)
pred_proba = pipeline.predict_proba(msg_test)
print classification_report(pred,label_test)
print accuracy_score(pred, label_test) # 95% accuracy
a = label_test == 'ham'
print roc_auc_score(a*1,pred_proba[:,0]) #auc 0.96

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:13:53 2018

@author: aananya
"""

# Natural Language Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
# to import all stopwords into spyder.
from nltk.corpus import stopwords
#importing stemming class-
from nltk.stem.porter import PorterStemmer
#corpus is a collection of text- an empty list
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    #stemming- taking the root of the word. Love: root for loving,loved,loves, will-love,,etc!
    #- to avoid sparcity 
    #also - we do it for words not the whole list
    ps = PorterStemmer()
    #using set so that the algo can execute faster!
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#for 1500 most common words.!
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
#800 to train , 200 to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
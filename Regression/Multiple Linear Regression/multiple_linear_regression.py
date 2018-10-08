#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  13 1:21:39 2018

@author: aananya

-- Multiple Linear Regression--

"""



#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data(using one hot encoder from sklearn)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# to Avoid the Dummy Variable Trap!!
X = X[:, 1:]

# Splitting 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Fitting to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting 
y_pred = regressor.predict(X_test)
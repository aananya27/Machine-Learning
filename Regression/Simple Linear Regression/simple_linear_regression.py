
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  12 12:13:28 2018

@author: aananya

--Simple Linear Regression--
"""



#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset- (Training and Test)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



# FitLR to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting

y_pred = regressor.predict(X_test)

# Visualising - Trining part
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#to finish and show the graph:
plt.show()

# Visualising - Test part
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
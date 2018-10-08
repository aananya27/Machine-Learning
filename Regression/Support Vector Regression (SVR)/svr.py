
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  17 2:13:45 2018

@author: aananya

--SVR--
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# no splitting, but Fs needed

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting with 6.5
y_pred = regressor.predict(6.5)
y_pred = sc_y.inverse_transform(y_pred)
'''

plt.scatter(X, y, color = 'orange')
plt.plot(X, regressor.predict(X), color = 'cyan')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''

# Visualising -better way
X_grid = np.arange(min(X), max(X), 0.01) #  0.01 instead of 0.1 step becase the data is fature scaled!
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'orange')
plt.plot(X_grid, regressor.predict(X_grid), color = 'cyan')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
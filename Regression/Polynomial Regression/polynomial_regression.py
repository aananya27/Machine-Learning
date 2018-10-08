
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:27:14 2018

@author: aananya
--Polynomial regression--
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# no need to split into train and test here!!

# Fitting LR to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Then Fitting Polynomial Reg
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualisin Lr results
plt.scatter(X, y, color = 'magenta')
plt.plot(X, lin_reg.predict(X), color = 'cyan')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

''' # Visualising Pr results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''

# Visualising PR with better resolutions- such a better curve.!
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'orange')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting -lr
lin_reg.predict(6.5)

# Predicting -pr! BETTER!!
lin_reg_2.predict(poly_reg.fit_transform(6.5))

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 5:35:11 2018

@author: aananya
--Decision Tree Regression--
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting 
y_pred = regressor.predict(6.5)

# Visualising 
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'yellow')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
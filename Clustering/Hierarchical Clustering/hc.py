
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 7:01:10 2018

@author: aananya
-- Hierarchical Clustering(HC) --
"""

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
#no need of making a 'y' here

#no split or fs required here!

# Making the dendrogram for finding optimal cluster number-
# see notes for how number of clusters happen, and how these work
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'yellow', label = 'Cl 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'pink', label = 'Cl 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'orange', label = 'Cl 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'magenta', label = 'Cl 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'purple', label = 'Cl 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income ')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
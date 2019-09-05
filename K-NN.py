# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:48:10 2019

@author: sarth
"""

#K-NN (K nearest neighbors)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('KNN')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

#splitting the dataset into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#Fitting classifier to training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #p = 2 is for euclidean distance
classifier.fit(X_train, y_train)

#Predicting results
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


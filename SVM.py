# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:19:01 2019

@author: sarth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data
dataset = pd.read_csv('SupportVectorMachines.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_set = .25, random_state = 0)

#Preprocessing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#Fitting the SVM classifier into the dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

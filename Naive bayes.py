# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:42:39 2019

@author: sarth
"""

#naive bayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Retrieving the datasset
dataset = pd.read_csv('naive_baeyes.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

#Train test split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_set = 0.25, random_state = 0)

#Preprocessing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#Fitting the naive bayes model to the data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the values
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
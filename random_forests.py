# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:41:19 2019

@author: sarth
"""


#Random Forests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Random_Forests.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

#Splitting data into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_set = 0.25, random_state = 0)

#Preprocessing data by feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#Fitting the model to the training data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the results
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

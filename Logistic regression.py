# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:35:25 2019

@author: sarth
"""

#Logistic regression 
import numpy as np
import pandas as pd
import matplotlib.pypolt as plt

dataset = pd.read_csv('Logisticregression.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

#Splitting dataset into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

# Creating confusion matrix to evaluate the performance of the LogisticRegressor
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

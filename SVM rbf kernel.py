# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:59:36 2019

@author: sarth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('SVM_kernel.csv')
X = dataset.iloc[:, [2,3]].value
y = dataset.iloc[:, -1].value

#train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_set = 0.2, random_state = 0)

#Preprocessing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

#fitting SVM kernel to dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

#Predicting results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
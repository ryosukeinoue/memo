# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:39:12 2018

@author: rinoue
"""

from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data[:,[2,3]]
y=iris.target
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train1=sc.transform(X_train)
X_test1=sc.transform(X_test)
import numpy as np
np.mean(X_test1,0)

from sklearn.linear_model import perceptron
ppn=perceptron.Perceptron(n_iter=100,eta0=0.1,shuffle=True)
ppn.fit(X_train1,y_train)
np.sum(ppn.predict(X_train1)==y_train)/len(y_train)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=10000)
lr.fit(X_train1,y_train)

np.sum(lr.predict(X_train1)==y_train)/len(y_train)

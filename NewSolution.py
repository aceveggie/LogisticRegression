'''
Program to test LogisticRegression using Sklearn toolkit
'''
from __future__ import division
import numpy as np
import scipy

from scipy import linalg as la
import math
from sklearn.linear_model import LogisticRegression
import Loaders

data, labels = Loaders.readFile("iris-data.txt",True)

lrclassfier = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1)
classifier = lrclassfier.fit(data, labels)
predicted_set = classifier.predict(data)

for i in zip(labels, predicted_set):
	print i[0],i[1],i[0]==i[1]

print 'Accuracy on training data: ',(np.sum(predicted_set == labels)/len(labels))*100,'%'
'''
Program to test LogisticRegression classifier
'''
from __future__ import division
import numpy as np
import scipy

import Loaders
import LogisticRegression
import Sigmoid

# from scipy import linalg as la
# import math
# from sklearn.linear_model import LogisticRegression


## load data
## add 1 column vector to entire data
## initialize parameters
## perform gradient descent
## {
## 	until number of iterations is satisfied
## 	calculate cost
## 	adjust parameters
## }
## predict using final parameters
## calculate accuracy

#############################################################################################################################
#################################### Multiclass or 3 class data classification#############################################################
#############################################################################################################################

data,labels = Loaders.readFile("iris-data.txt",True)
m,n = data.shape
# lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 8000, regularized=True, normalization = 'l1')
lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 8000, regularized=True, normalization = 'l2')
learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))
print 'Number of classes', len(np.unique(labels))
print 'learntParameters(one per class): ',learntParameters
print 'final_costs: ', final_costs
#print len(learntParameters)
classifedLabels = []
for eachData in data:
	classifedLabels.append(lr1.classify(eachData, learntParameters))
classifedLabels = np.array(classifedLabels)

# print 'original label', 'classifedLabels'
# for each in zip(labels, classifedLabels):
# 	print each[0],', ', each[1],', ', each[0]==each[1]

print 'Accuracy on training data: ',(np.sum(classifedLabels == labels)/len(labels))*100,'%'

#############################################################################################################################
#################################### 2 class data classification#############################################################
#############################################################################################################################

data,labels = Loaders.readFile("mod-iris.txt",True)
m,n = data.shape
lr1 = LogisticRegression.LogisticRegression(data, labels, 2.0, 100)
learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))
print 'Number of classes', len(np.unique(labels))
print 'learntParameters(only 1 learnt parameter): ',learntParameters
print 'final_costs: ', final_costs
classifedLabels = []
for eachData in data:
	classifedLabels.append(lr1.classify(eachData, learntParameters))
classifedLabels = np.array(classifedLabels)
# print 'original label', 'classifedLabels'
# for each in zip(labels, classifedLabels):
# 	print each[0],', ', each[1],', ', each[0]==each[1]

print (np.sum(classifedLabels == labels)/len(labels))*100,'%'
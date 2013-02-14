from __future__ import division
import numpy as np
import scipy

import Loaders
import LogisticRegression
import Sigmoid

# load data
# add 1 column vector to entire data
# initialize parameters
# perform gradient descent
# {
# 	until number of iterations is satisfied
# 	calculate cost
# 	adjust parameters
# }
# predict using final parameters
# calculate accuracy

data,labels = Loaders.readFile("iris-data.txt",True)
m,n = data.shape
# initialize theta values to zero
# train

lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 2000)

learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))

classifedLabels = []
for eachData in data:
	classifedLabels.append(lr1.classify(eachData, learntParameters))
classifedLabels = np.array(classifedLabels)
print 'original label', 'classifedLabels'
for each in zip(labels, classifedLabels):
	print each[0],', ', each[1],', ', each[0]==each[1]

print (np.sum(classifedLabels == labels)/len(labels))*100,'%'
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
#lr1 = LogisticRegression.LogisticRegression(data, labels)
lr1 = LogisticRegression.LogisticRegression(data, labels, 1.0, 2000)

learntParameters, final_costs = lr1.train(data, labels, np.unique(labels))
for i in zip(learntParameters, final_costs):
	print i[0].T,'\t', i[1]

classifedLabel = []
for eachData in data:
	classifedLabel.append(lr1.classify(eachData, learntParameters))
classifedLabel = np.array(classifedLabel)

print (np.sum(classifedLabel == labels)/len(labels))*100,'%'
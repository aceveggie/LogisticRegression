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

# load data and add column vector of 1's
data,labels = Loaders.readFile("iris-data.txt",True)
m,n = data.shape
# initialize theta values to zero
theta_init = np.zeros((n,1))
# train
lr1 = LogisticRegression.LogisticRegression(data, labels)
from __future__ import division
import numpy as np
import scipy

import Loaders
import LogisticRegression

data,labels = Loaders.readFile("iris-data.txt")
# print data
# print '--'
# print labels
# print '--'
# print np.where(labels ==1)
# print data[np.where(labels ==1)]

m,n = data.shape
gd = np.ones((m,1.00))
gd = np.column_stack((gd, data))
# print gd.shape
# print gd

theta_init = np.zeros((n+1,1))

lr1 = LogisticRegression.LogisticRegression(theta_init, gd, labels)

#classifiedlabels = lr1.classify()
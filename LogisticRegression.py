from __future__ import division
import numpy as np
import scipy
import Loaders
import math
# import bigfloat

class LogisticRegression:
	def __init__(self, init_theta, data, labels, num_iters = 100):
		self.train(init_theta, data, labels, num_iters = 100)
	def train(self, init_theta, data, labels, num_iters = 100):
		'''
		train the classifier. One classifier per unique label
		'''
		n_labels = np.unique(labels)

		# for eachIter in range(num_iters):
		# 	self.gradientDescent(init_theta, data, labels, alpha = 0.01, num_iters = 100)
		# in a loop
		# compute cost
		# run gradient descent to change theta values

		pass
	def classify(self, init_theta, data):
		'''
		classify given data and return a list of associated classified labels
		'''
		return classifiedlabels

	def sigmoidCalc(self, data):
		'''
		calculate the sigmoid of the given data
		'''
		data = np.array(data, dtype = np.longdouble)
		g = np.zeros(data.shape, dtype = np.float64)
		m,n = data.shape
		#print data.dtype

		
		for eachRow in range(m):
			for eachCol in range(n):
				print float(np.exp(-data[eachRow, eachCol]))
				g[eachRow,eachCol] = (1.0) / (1.0 + (math.exp(-data[eachRow, eachCol])))
				
		return g

	def gradientDescent(self, init_theta, data, labels, alpha = 0.01, num_iters=100):
		'''

		'''
		m = len(labels)
		nRows, nCols = data.shape
		cost_history = []
		temp = np.zeros((1,nCols))

		# print data.shape
		# print init_theta.shape
		for eachIter in range(num_iters):
			for i in range(nCols):

				np.dot(data, init_theta)
				temp[0,i] = init_theta[i,:] - (alpha/m) * (np.sum(((np.dot(data, init_theta))-labels) * data[:,i]))

			for i in range(nCols):
				init_theta[i,:] = temp[0,i]

			cost, grad = self.computeCost(data, labels, init_theta)
			print "cost:", cost, 'at iteration',eachIter
			cost_history.append(cost)
		return theta, cost

	def computeCost(self, data, labels, init_theta):
		'''
		compute cost of the given value of theta and return it
		'''
		m = len(labels)
		grad = np.zeros(init_theta.shape)

		#J = (-1.0/ m) * ( sum( log(sigmoid(X * theta)) .* y + ( log ( 1- sigmoid(X * theta) ) .* ( 1 - y ) )) );

		J = (-1.0/m) * (np.sum( (np.log(np.dot(data, init_theta))) * labels + ( np.log( 1-self.sigmoidCalc(np.dot(data, init_theta) ) ) * labels ) ))

		nRows, nCols = data.shape

		d = self.sigmoidCalc( np.dot(data, init_theta)) - labels;
		R = np.zeros(data.shape)
		
		print d.shape
		print data.shape
		
		R = d * data

		cost = J
		gradient = np.dot((1.0/m), np.sum(R, 0))

		return cost, gradient

class GradientDescent(object):
	"""docstring for Gradient
	Descent"""
	def __init__(self, arg):
		super(GradientDescent, self).__init__()
		self.arg = arg
		
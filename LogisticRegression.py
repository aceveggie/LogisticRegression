from __future__ import division
import numpy as np
import scipy
import Loaders
import math
# import bigfloat

class LogisticRegression:
	def __init__(self, data, labels, regularized= False, num_iters = 100):
		'''
		constructor just takes data and labels
		'''
		unique_classes = np.unique(labels)
		self.train(data, labels, regularized, unique_classes, num_iters)
		pass


	def train(self, data, Olabels, regularized, unique_classes, num_iters):
		'''
		train the classifier. One classifier per unique label
		'''
		#print Olabels
		m,n = data.shape

		# map labels to program friendly labels
		labels = np.zeros(Olabels.shape)
		uniq_Olabel_names = np.unique(Olabels)
		uniq_label_list = range(len(uniq_Olabel_names))

		for each in zip(uniq_Olabel_names, uniq_label_list):
			o_label_name = each[0]
			new_label_name = each[1]
			labels[np.where(Olabels == o_label_name)] = new_label_name


		# now labels variable contains labels starting from 0 to (num_classes -1)
		num_classes = len(unique_classes)

		Init_Thetas = [] # to hold initial values of theta
		
		Thetas = [] # to hold final values of theta to return
		
		Cost_Thetas = [] # cost associated with each theta
		
		Cost_History_Theta = [] # contains list of varying cost thetas
		
		# if num_classes = 2, then N_Thetas will contain only 1 Theta
		# if num_classes >2, then N_Thetas will contain num_classes number of Thetas.
		
		if(num_classes == 2):
			theta_init = np.zeros((n,1))
			Init_Thetas.append(theta_init)
 		else:
 			for eachInitTheta in range(num_classes):
 				theta_init = np.zeros((n,1))
 				Init_Thetas.append(theta_init)

 		# print 'num_classes', num_classes
 		# print 'init thetas', Init_Thetas

 		for eachClass in range(num_classes):
 			# load data local of the init_theta
 			# +ve class is 1 and rest are zeros
 			# its a one vs all classifier

 			local_labels = np.zeros(labels.shape)

 			
 			local_labels[np.where(labels == eachClass)] = 1
 			print local_labels

 			# assert to make sure that its true
 			assert(len(np.unique(local_labels)) == 2)
 			assert(len(local_labels) == len(labels))

 			
			init_theta = Init_Thetas[eachClass]

			new_theta, cost_theta, cost_history = self.gradientDescent(init_theta, data, local_labels, regularized, num_iters)
			
			# Thetas.append(new_theta)
			# Cost_Thetas.append(cost_theta)
			# Cost_History_Theta.append(cost_history)
		return Thetas
	

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
				#print float(np.exp(-data[eachRow, eachCol]))
				g[eachRow,eachCol] = (1.0) / (1.0 + (math.exp(-data[eachRow, eachCol])))
				
		return g


	def gradientDescent(self, init_theta, data, labels, regularized, num_iters = 100, alpha = 0.01):
		'''
		for given number of iterations:
			perform gradient descent and simultaneously compute cost
		arrive at final theta
		'''
		y = labels
		X = data
		#grad = x * (y- sigmoid(theta'*x))';

		print 'X: ', X.shape
		print 'init_theta: ', init_theta.shape
		for eachIteration in range(num_iters):
			# computer cost of the theta	
			print 'cost at iteration: ', eachIteration, self.computeCost(data, labels, regularized, init_theta)	
			# adjust parameters using gradient descent
			# continue until number of iterations is satisfied
			gradientVal = np.dot (data, (y - self.sigmoidCalc( np.dot(X, init_theta) ) ) )
			init_theta = init_theta - (alpha/m) * gradientVal
		return init_theta


	def computeCost(self, X, y, regularized, init_theta):
		'''
		compute cost of the given value of theta and return it
		'''
		if(regularized == True):
			llambda = 1
		else:
			llambda = 0

		theta = init_theta

		m, n = X.shape
		theta1 = init_theta[0]
		theta2 = init_theta[1:len(init_theta)]
		# select first column of the data since it was manually added as an extra field while loading the data (Loaders.py).
		X1 = X[:,0]
		X2 = X[:,1:n]

		A = np.log( self.sigmoidCalc(np.dot( X, theta ))) * (-y)  
		b1 = 1-y
		b2 = np.log(1 - self.sigmoidCalc(np.dot(X, theta)))
		B = b1 * b2

		regularized_parameter = np.dot( (llambda/2*m), np.sum( np.power(theta2, 2)))

		cost =  ( (1/m) * np.sum(A-B))  + regularized_parameter


		return cost
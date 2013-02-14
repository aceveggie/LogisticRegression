from __future__ import division
import numpy as np
import scipy
import Loaders
import math
from scipy.optimize import fmin_bfgs
# import bigfloat

class LogisticRegression:
	def __init__(self, data, labels, alpha, num_iters, regularized= False):
		'''
		constructor just takes data and labels
		'''
		# self.data = data
		# self.labels = labels
		self.num_iters = num_iters
		self.alpha = alpha

		#self.train(data, labels, unique_classes, regularized, num_iters)
		pass


	def train(self, data, Olabels, unique_classes, regularized=False):
		'''
		train the classifier. One classifier per unique label
		'''

		num_iters = self.num_iters
		m,n = data.shape

		# map labels to program friendly labels
		labels = np.zeros(Olabels.shape)
		uniq_Olabel_names = np.unique(Olabels)
		uniq_label_list = range(len(uniq_Olabel_names))

		for each in zip(uniq_Olabel_names, uniq_label_list):
			o_label_name = each[0]
			new_label_name = each[1]
			labels[np.where(Olabels == o_label_name)] = new_label_name

		labels = labels.reshape((len(labels),1))
		# now labels variable contains labels starting from 0 to (num_classes -1)
		#print unique_classes
		num_classes = len(unique_classes)

		Init_Thetas = [] # to hold initial values of theta
		
		Thetas = [] # to hold final values of theta to return
		
		Cost_Thetas = [] # cost associated with each theta
		
		Cost_History_Theta = [] # contains list of varying cost thetas
		
		# if num_classes = 2, then N_Thetas will contain only 1 Theta
		# if num_classes >2, then N_Thetas will contain num_classes number of Thetas.
		#print np.unique(labels)
		if(num_classes == 2):
			init_theta = np.zeros((n,1))
			Init_Thetas.append(init_theta)
			## yet to be finished currently not working
			# new_theta, final_cost = self.computeGradient(data, labels, init_theta, self.alpha, self.num_iters)
			# #print final_cost
			# Thetas.append(new_theta)
			# Cost_Thetas.append(final_cost)
 		elif(num_classes>2):
 			for eachInitTheta in range(num_classes):
 				theta_init = np.zeros((n,1))
 				Init_Thetas.append(theta_init)
 				pass

	 		for eachClass in range(num_classes):
	 			# load data local of the init_theta
	 			# +ve class is 1 and rest are zeros
	 			# its a one vs all classifier

	 			local_labels = np.zeros(labels.shape)

	 			
	 			local_labels[np.where(labels == eachClass)] = 1
	 		

	 			# assert to make sure that its true
	 			assert(len(np.unique(local_labels)) == 2)
	 			assert(len(local_labels) == len(labels))
	 			# print eachClass
	 			# print Init_Thetas
				init_theta = Init_Thetas[eachClass]

				
				new_theta, final_cost = self.computeGradient(data, local_labels, init_theta, self.alpha, self.num_iters)
				#print final_cost
				Thetas.append(new_theta)
				Cost_Thetas.append(final_cost)
			
		return Thetas, Cost_Thetas
	

	def classify(self, data, Thetas):
		'''
		classify given data and return a list of associated classified labels
		'''
		# since it is a one vs all classifier, load all classifiers and pick most likely
		# i.e. which gives max value for sigmoid(X*theta)
		mvals = []
		for eachTheta in Thetas:
			mvals.append(self.sigmoidCalc(np.dot(data, eachTheta)))
			pass
		return mvals.index(max(mvals))+1


	
	def sigmoidCalc(self, data):
		'''
		calculate the sigmoid of the given data
		'''
		# if(len(data.flatten()) == 1 ):
		# 	data = data.reshape((1,1))
		
		data = np.array(data, dtype = np.longdouble)
		#g = np.zeros(data.shape, dtype = np.float64)
		g = 1/(1+np.exp(-data))
		
		return g

	def computeCost(self,data, labels, init_theta, regularized=False):
		'''
		compute cost of the given value of theta and return it
		'''
		if(regularized == True):
			llambda = 1
		else:
			llambda = 0

		m,n = data.shape
		
		J = 0;

		grad = np.zeros(init_theta.shape);

		theta2 = init_theta[range(1,init_theta.shape[0]),:]
		
		regularized_parameter = np.dot(llambda/(2*m), np.sum( theta2 * theta2))
		
		J = (-1.0/ m) * ( np.sum( np.log(self.sigmoidCalc( np.dot(data, init_theta))) * labels + ( np.log ( 1 - self.sigmoidCalc(np.dot(data, init_theta)) ) * ( 1 - labels ) )));
		
		J = J + regularized_parameter;

		return J

	def computeGradient(self,data, labels, init_theta, alpha, num_iters = 100, regularized=False):
		m,n = data.shape

		if(regularized == True):
			llambda = 1
		else:
			llambda = 0
		
		for eachIteration in range(num_iters):
			cost = self.computeCost(data, labels, init_theta);
			
			#compute gradient
			
			B = self.sigmoidCalc(np.dot(data, init_theta) - labels)
			
			A = (1/m)*np.transpose(data)
			
			grad = np.dot(A,B)
			
			
			A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels )
			B =  data[:,0].reshape((data.shape[0],1))
			
			grad[0] = (1/m) * np.sum(A*B)
			
			A = (self.sigmoidCalc(np.dot(data, init_theta)) - labels)
			B = (data[:,range(1,n)])
			
			for i in range(1, len(grad)):
				A = (self.sigmoidCalc(np.dot(data,init_theta)) - labels )
				B = (data[:,i].reshape((data[:,i].shape[0],1)))
				grad[i] = (1/m)*np.sum(A*B) + ((llambda/m)*init_theta[i])

						
			init_theta = init_theta - (np.dot((alpha/m), grad))
			
		return init_theta, cost
from __future__ import division
import numpy as np
import scipy
import math

def sigmoidCalc(data):
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
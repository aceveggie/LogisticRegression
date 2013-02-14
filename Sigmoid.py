from __future__ import division
import numpy as np
import scipy
import math

def sigmoidCalc(data):
	'''
	calculate the sigmoid of the given data
	'''
	g = 1.0 / (1.0 + np.exp(-data))	
	return g
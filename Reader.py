import numpy as np
import Loaders

data,labels = Loaders.readFile("iris.data.txt")

for eachDataEntry in zip(data, labels):
	print eachDataEntry[0], eachDataEntry[1]
	print '--'
print np.unique(labels)
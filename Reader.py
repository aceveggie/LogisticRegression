import numpy as np
import Loaders
import LogisticRegression

data,Olabels = Loaders.readFile("iris-data.txt")

m,n = data.shape

# map labels to program friendly labels
labels = np.zeros(Olabels.shape)

unique_classes = np.unique(Olabels)
# map labels to user program understandable format
Label_Map = {}
label_id = 0
for eachClass in unique_classes:
	# map unique label strings to labels starting from 0 to (num_classes - 1)
	Label_Map[eachClass] = label_id
	labels[ np.where(Olabels == eachClass) ] = Label_Map[eachClass]
	label_id += 1

num_classes = len(unique_classes)

for each in zip(data, Olabels, labels):
	print each
lr = LogisticRegression.LogisticRegression(data, Olabels)
print lr.sigmoidCalc(data)
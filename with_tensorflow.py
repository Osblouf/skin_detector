import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Load and format training data
data = np.genfromtxt("Skin_NonSkin.txt")
np.random.shuffle(data)
labels = data[:, 3] - 1
inputs = data[:, 0:3]/255
# richer_inputs = np.concatenate((inputs, inputs**2), axis=1)
richer_inputs = inputs
dataSetSize = data.shape[0]
trainingSize = int(round(dataSetSize * 0.6, 0))
validateSize = int(round(dataSetSize * 0.2, 0))
testSize = dataSetSize - trainingSize - validateSize
X = richer_inputs[0:trainingSize, :]
y = labels[0:trainingSize]
Xval = richer_inputs[trainingSize:(trainingSize+validateSize), :]
yval = labels[trainingSize:(trainingSize+validateSize)]
Xtest = richer_inputs[(trainingSize+validateSize):dataSetSize, :]
ytest = labels[(trainingSize+validateSize):dataSetSize]

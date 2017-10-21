# Skin categorization
# Data from https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

# import numpy as np
from simpleLogisticRegression import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc

# Load data from text file
data = np.genfromtxt("Skin_NonSkin.txt")
print("Data shape :")
print(data.shape)

# Randomize data and set truth table from 0 to 1
np.random.shuffle(data)
data[:, 3] = data[:, 3] - 1
data[:, 0:3] = data[:, 0:3]/255
print("Displaying some data :")
print(data[0:5, :])

# Separate training samples, validation samples and test samples
dataSetSize = data.shape[0]
trainingSize = int(round(dataSetSize * 0.6, 0))
validateSize = int(round(dataSetSize * 0.2, 0))
testSize = dataSetSize - trainingSize - validateSize
print("Training set size :")
print(trainingSize)
print("Validation set size :")
print(validateSize)
print("Test set size :")
print(testSize)
X = data[0:trainingSize, 0:3]
y = data[0:trainingSize, 3]
Xval = data[trainingSize:(trainingSize+validateSize), 0:3]
yval = data[trainingSize:(trainingSize+validateSize), 3]
Xtest = data[(trainingSize+validateSize):dataSetSize, 0:3]
ytest = data[(trainingSize+validateSize):dataSetSize, 3]

# Plot some data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Yyes = np.where(yval[0:10000] == 1)[0]
ax.scatter(Xval[Yyes, 0], Xval[Yyes, 1], Xval[Yyes, 2], c='g', marker='o')
Yno = np.where(yval[0:10000] == 0)[0]
ax.scatter(Xval[Yno, 0], Xval[Yno, 1], Xval[Yno, 2], c='b', marker='^')
plt.show()

# Test gradient algorithms a loop
for index in range(200):
    theta = np.random.rand(4, 1) - 0.5
    back_prop = back_propagation_unregularized(theta, *(Xval[0:64], yval[0:64]))
    numerical = numerical_gradient_unregularized(theta, Xval[0:64], yval[0:64])
    error = np.absolute(back_prop - numerical)
    if np.max(error) >= 0.000000001:
        print("NOK")

# Setting the model (logistic regression)
theta = np.random.rand(4, 1) - 0.5
print("*******\nCost function :")
print(cost_function_unregularized(theta, *(X[0:10000, :], y[0:10000])))
theta_trained = train_logistic_regression(theta, X[0:10000, :], y[0:10000])
print("Cast function on validation test :")
print(cost_function_unregularized(theta_trained, *(Xval[0:5000, :], yval[0:5000])))

# Try with new features (add squared features)
Xsquare = X**2
Xnew = np.concatenate((X, Xsquare), axis=1)

XvalSquare = Xval**2
Xvalnew = np.concatenate((Xval, XvalSquare), axis=1)

XtestSquare = Xtest**2
Xtestnew = np.concatenate((Xtest, XtestSquare), axis=1)

# Setting the new model (logistic regression)
print("*******\nCost function :")
theta = np.random.rand(7, 1) - 0.5
print(cost_function_unregularized(theta, *(Xnew[0:10000, :], y[0:10000])))
theta_trained = train_logistic_regression(theta, Xnew[0:10000, :], y[0:10000])
print("Cast function on validation test :")
print(cost_function_unregularized(theta_trained, *(Xvalnew[0:5000, :], yval[0:5000])))

# Using regularization
lambda_set = [0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
cost = np.zeros(len(lambda_set))
theta = np.random.rand(7, 1) - 0.5
for index, lam in enumerate(lambda_set):
    theta_temp = train_regularized_logistic_regression(theta, Xvalnew[0:10000, :], yval[0:10000], lam)
    cost[index] = cost_function_regularized(theta_temp, *(Xvalnew[0:10000, :], yval[0:10000], lam))

used_lambda = lambda_set[np.argmin(cost)]
print("*******\nRegularized cost function :")
print(cost_function_regularized(theta, *(Xnew[0:10000, :], y[0:10000], used_lambda)))
theta_trained = train_regularized_logistic_regression(theta, Xnew[0:10000, :], y[0:10000], used_lambda)
print("Cast function on validation test :")
print(cost_function_regularized(theta_trained, *(Xvalnew[0:5000, :], yval[0:5000], used_lambda)))
print("Used lambda :")
print(used_lambda)

print("*******\nUsing mini batch :")
cost = np.zeros(len(lambda_set))
theta = np.random.rand(7, 1) - 0.5
for index, lam in enumerate(lambda_set):
    theta_temp = train_with_mini_batch(theta, Xvalnew, yval, lam)
    cost[index] = mini_batch_cost_function(theta_temp, *(Xvalnew, yval, lam))
used_lambda = lambda_set[np.argmin(cost)]
print(mini_batch_cost_function(theta, *(Xnew, y, used_lambda)))
theta_trained = train_with_mini_batch(theta, Xnew, y, used_lambda)
print("Cost function on validation test :")
print(mini_batch_cost_function(theta_trained, *(Xvalnew, yval, used_lambda)))
print("Used lambda :")
print(used_lambda)

# Predictions on test set :
predictions = predict(theta_trained, Xtestnew[0:10000, :])
true_positives_index = np.where((ytest[0:10000] == 1) & (predictions >= 0.5))
false_positives_index = np.where((ytest[0:10000] == 0) & (predictions >= 0.5))
true_negatives_index = np.where((ytest[0:10000] == 0) & (predictions < 0.5))
false_negatives_index = np.where((ytest[0:10000] == 1) & (predictions < 0.5))
true_positives_nbr = ytest[true_positives_index].shape[0]
false_positives_nbr = ytest[false_positives_index].shape[0]
true_negatives_nbr = ytest[true_negatives_index].shape[0]
false_negatives_nbr = ytest[false_negatives_index].shape[0]
precision = true_positives_nbr / (false_positives_nbr + true_positives_nbr)
recall = true_positives_nbr / (true_positives_nbr + false_negatives_nbr)

print(true_positives_nbr + false_positives_nbr + false_negatives_nbr + true_negatives_nbr)
print("Recall :")
print(recall)
print("Precision :")
print(precision)
print("True positives :")
print(true_positives_nbr)
print("False positives :")
print(false_positives_nbr)
print("False negatives :")
print(false_negatives_nbr)

# Visualize some results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xtestnew[true_positives_index, 0],
           Xtestnew[true_positives_index, 1],
           Xtestnew[true_positives_index, 2],
           c='b', marker='o')
ax.scatter(Xtestnew[true_negatives_index, 0],
           Xtestnew[true_negatives_index, 1],
           Xtestnew[true_negatives_index, 2],
           c='g', marker='^')
ax.scatter(Xtestnew[false_positives_index, 0],
           Xtestnew[false_positives_index, 1],
           Xtestnew[false_positives_index, 2],
           c='r', marker='o')
ax.scatter(Xtestnew[false_negatives_index, 0],
           Xtestnew[false_negatives_index, 1],
           Xtestnew[false_negatives_index, 2],
           c='black', marker='^')
plt.show()

# Test it on image :
print("Display test image")
image = misc.imread("test2.jpg", mode='RGB')
plt.imshow(image)
plt.show()
print("Predict on this image :")
flatten_image = image.flatten().reshape((image.shape[0]*image.shape[1], 3))/255
flatten_image[:, (0, 2)] = flatten_image[:, (2, 0)]
flatten_image_new_features = flatten_image**2
new_flatten_image = np.concatenate((flatten_image, flatten_image_new_features), axis=1)
mask = mini_batch_prediction(theta_trained, new_flatten_image)
masked_image = np.zeros(flatten_image.shape)
for index in np.arange(0, flatten_image.shape[0], 10000):
    flatten_image[index:index+10000, 0] = flatten_image[index:index+10000, 0] * mask[index:index+10000, 0]
    flatten_image[index:index+10000, 1] = flatten_image[index:index+10000, 1] * mask[index:index+10000, 0]
    flatten_image[index:index+10000, 2] = flatten_image[index:index+10000, 2] * mask[index:index+10000, 0]

flatten_image[:, (0, 2)] = flatten_image[:, (2, 0)]
deflated_image = flatten_image.flatten().reshape((image.shape[0], image.shape[1], 3))
plt.imshow(deflated_image)
plt.show()

print("End.")

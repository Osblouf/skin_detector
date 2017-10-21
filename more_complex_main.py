# This is a try to complexify the previous model

from LogisticRegression import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc

# Init parameters
# theta = np.random.rand((7*2)+3, 1) * 0.5
# theta = np.random.rand((7*3)+4, 1) * 0.5
theta = np.random.rand((7*1)+2, 1) * 0.5

# Load and format training data
data = np.genfromtxt("Skin_NonSkin.txt")
np.random.shuffle(data)
labels = data[:, 3] - 1
inputs = data[:, 0:3]/255
richer_inputs = np.concatenate((inputs, inputs**2), axis=1)
#richer_inputs = inputs
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

print("Testing gradient computation :")
for test in range(0, 100):
    #    theta = np.random.rand((7*3)+4, 1) * 0.5
    # theta = np.random.rand((7*2)+3, 1) * 0.5
    theta = np.random.rand((7*1)+2, 1) * 0.5
    args = (X[0:10000, :], y[0:10000], 0.01)
    numerical = numerical_gradient(theta, *args)
    back_prop_test = back_prop(theta, *args)
    error = np.absolute(back_prop_test - numerical)
    if np.max(error) >= 0.000000001:
        print("NOK In test : " + str(test))
        print("\tError = " + str(error))
print("Mini batch cost and cost difference = " + str(mini_batch_cost(theta, *args) - cost_function(theta, *args)))
temp1 = back_prop(theta, *args)
temp2 = mini_batch_back_prop(theta, *args)
print("Mini batch gradient - gradient difference = " + str(temp1 - temp2))

# lambda_list = [0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6]
lambda_list = [0, 0.001, 0.003, 0.006]
costs = np.zeros((len(lambda_list), 1))
print("Finding best lambda to use :")
for index, lam in enumerate(lambda_list):
    print("\tTrying with lambda = " + str(lam))
    theta_trained = train_model(theta, *(Xval, yval, lam), max_iteration=5000)
    costs[index] = cost_function(theta_trained, *(Xval, yval, lam))
# optimum_lambda = 0.003
optimum_lambda = lambda_list[np.argmin(costs)]
print("Lambda to use : " + str(optimum_lambda))

print("Show some data : ")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Yyes = np.where(yval[0:10000] == 1)[0]
ax.scatter(Xval[Yyes, 0], Xval[Yyes, 1], Xval[Yyes, 2], c='g', marker='o')
Yno = np.where(yval[0:10000] == 0)[0]
ax.scatter(Xval[Yno, 0], Xval[Yno, 1], Xval[Yno, 2], c='b', marker='^')
plt.show()

print("Train the model on all the data : ")
# theta = np.random.rand((7*3)+4, 1) * 0.5
theta = np.random.rand((7 * 1) + 2, 1) * 0.5
trained_theta = train_model(theta, *(X, y, optimum_lambda), max_iteration=10000)
np.save("Trained_theta_3_features_1_node.npy", trained_theta)
trained_theta = np.load("Trained_theta_3_features_1_node.npy")
print(trained_theta)

# Predictions on test set :
ytest = np.reshape(ytest, (ytest.shape[0], 1))
a1, z1, a2, predictions = predict(trained_theta, Xtest[0:10000, :])
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
ax.scatter(Xtest[true_positives_index, 0],
           Xtest[true_positives_index, 1],
           Xtest[true_positives_index, 2],
           c='b', marker='o')
ax.scatter(Xtest[true_negatives_index, 0],
           Xtest[true_negatives_index, 1],
           Xtest[true_negatives_index, 2],
           c='g', marker='^')
ax.scatter(Xtest[false_positives_index, 0],
           Xtest[false_positives_index, 1],
           Xtest[false_positives_index, 2],
           c='r', marker='o')
ax.scatter(Xtest[false_negatives_index, 0],
           Xtest[false_negatives_index, 1],
           Xtest[false_negatives_index, 2],
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
# new_flatten_image = flatten_image
a1, z1, a2, mask = predict(trained_theta, new_flatten_image)
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
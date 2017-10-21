# This is the necessary tools for logistic regression
import numpy as np
from scipy import optimize


# THE sigmoid function
def sigmoid(matrix):
    return 1.0/(1.0+np.exp(-matrix))


# The sigmoid's derivative
def sigmoid_prime(matrix):
    return sigmoid(matrix)*(1-sigmoid(matrix))


# Prediction with forward prop
def predict(theta, inputs):
    m = inputs.shape[0]
    a = np.insert(inputs, [0], [1], axis=1)
    z = a.dot(theta).reshape((m, 1))
    return np.ndarray.flatten(sigmoid(z))


# Mini batch prediction
def mini_batch_prediction(theta, inputs):
    mini_batch_size = 1000
    data_set_size = inputs.shape[0]
    prediction = np.zeros((data_set_size, 1))
    mini_batch_nbr = int(data_set_size/mini_batch_size)
    for index in range(0, mini_batch_nbr):
        lower_index = mini_batch_size * index
        upper_index = lower_index + mini_batch_size
        prediction[lower_index:upper_index, 0] = predict(theta, inputs[lower_index:upper_index, :])
    lower_index = mini_batch_nbr*mini_batch_size
    upper_index = data_set_size
    prediction[lower_index:upper_index, 0] = predict(theta, inputs[lower_index:upper_index, :])
    return np.around(prediction)


# Unregularized cost function
def cost_function_unregularized(theta, *args):
    x, y = args
    m = x.shape[0]
    y = y.reshape((m, 1))  # Careful of y dimension !!
    a = np.insert(x, [0], [1], axis=1)
    z = a.dot(theta).reshape((m, 1))
    h = sigmoid(z)

    sum_costs = np.sum((-y)*np.log(h) - (1-y)*np.log(1-h))
    return (1/m) * sum_costs


# Regularized cost function
def cost_function_regularized(theta, *args):
    x, y, lambda_set = args
    return cost_function_unregularized(theta, *(x, y)) + (lambda_set/(2*x.shape[0]))*sum(theta**2)


# Mini batch cost function
def mini_batch_cost_function(theta, *args):
    x, y, lambda_set = args
    cost = 0
    mini_batch_size = 10000
    data_set_size = x.shape[0]
    mini_batch_nbr = int(data_set_size/mini_batch_size)
    for index in range(0, mini_batch_nbr):
        low_index = index*mini_batch_size
        upper_index = low_index + mini_batch_size
        x_temp = x[low_index:upper_index, :]
        y_temp = y[low_index:upper_index]
        cost = cost + cost_function_regularized(theta,
                                                *(x_temp, y_temp, lambda_set))
    low_index = mini_batch_nbr * mini_batch_size
    upper_index = data_set_size
    cost = cost + \
        cost_function_regularized(theta,
                                  *(x[low_index: upper_index, :],
                                    y[low_index: upper_index],
                                    lambda_set))
    return cost/(mini_batch_nbr+1)


# Unregularized back propagation
def back_propagation_unregularized(theta, *args):
    x, y = args
    m = x.shape[0]
    y = y.reshape((m, 1))  # Careful of y dimension !!
    a = np.insert(x, [0], [1], axis=1)
    h = sigmoid(a.dot(theta).reshape((m, 1)))
    delta = h - y
    big_delta = np.transpose(a).dot(delta)
    big_delta = np.ndarray.flatten(big_delta)
    return (1/m)*big_delta


# Regularize back prop !
def back_propagation_regularized(theta, *args):
    x, y, lambda_set = args
    lambda_theta = np.ones((theta.shape[0], 1)) * (lambda_set/x.shape[0])
    lambda_theta[0] = 0
    return back_propagation_unregularized(theta, *(x, y))


# Mini batch gradient
def mini_batch_gradient(theta, *args):
    x, y, lambda_set = args
    grad = np.zeros(theta.shape)
    mini_batch_size = 10000
    data_set_size = x.shape[0]
    mini_batch_nbr = int(data_set_size / mini_batch_size)
    for index in range(0, mini_batch_nbr):
        low_index = index * mini_batch_size
        upper_index = low_index + mini_batch_size
        x_temp = x[low_index:upper_index, :]
        y_temp = y[low_index:upper_index]
        grad = grad + \
            back_propagation_regularized(theta,
                                         *(x_temp, y_temp, lambda_set))
    low_index = mini_batch_nbr * mini_batch_size
    upper_index = data_set_size
    grad = grad + back_propagation_regularized(theta,
                                               *(x[low_index: upper_index, :],
                                                 y[low_index: upper_index],
                                                 lambda_set))
    return grad / (mini_batch_nbr + 1)


# Numerical unregularized gradient
def numerical_gradient_unregularized(theta, x, y):
    epsilon = 0.0001
    args = (x, y)
    grad = np.zeros((4, 1))
    plus_epsilon = np.identity(theta.shape[0]) * epsilon
    theta_minus_epsilon = theta.dot(np.ones((1, theta.shape[0])))
    theta_plus_epsilon = theta_minus_epsilon + plus_epsilon
    theta_minus_epsilon = theta_minus_epsilon - plus_epsilon
    for nbr in range(0, theta.shape[0]):
        theta_temp1 = theta_minus_epsilon[:, nbr]
        theta_temp2 = theta_plus_epsilon[:, nbr]
        cost1 = cost_function_unregularized(theta_temp1, *args)
        cost2 = cost_function_unregularized(theta_temp2, *args)
        grad[nbr, 0] = (cost2 - cost1)/(2*epsilon)

    grad = np.ndarray.flatten(grad)
    return grad


# Train the network
def train_logistic_regression(theta, x, y):
    max_training_loop = 10000
    res1 = optimize.fmin_cg(cost_function_unregularized,
                            theta,
                            fprime=back_propagation_unregularized,
                            args=(x, y),
                            maxiter=max_training_loop)
    return res1


# Train with regularized functions
def train_regularized_logistic_regression(theta, x, y, lam):
    max_training_loop = 10000
    res1 = optimize.fmin_cg(cost_function_regularized,
                            theta,
                            fprime=back_propagation_regularized,
                            args=(x, y, lam),
                            maxiter=max_training_loop)
    return res1


# Train with mini batch
def train_with_mini_batch(theta, x, y, lam):
    max_training_loop = 10000
    res1 = optimize.fmin_cg(mini_batch_cost_function,
                            theta,
                            fprime=mini_batch_gradient,
                            args=(x, y, lam),
                            maxiter=max_training_loop)
    return res1

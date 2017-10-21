import numpy as np
from scipy import optimize


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


def sigmoid_prime(matrix):
    return matrix * (1 - matrix)


def extract_theta(theta):
    # theta1 = theta[0:7 * 3]
    theta1 = theta[0:7 * 1]
    # theta1 = theta1.reshape((7, 3))
    theta1 = theta1.reshape((7, 1))
    # theta2 = theta[7 * 3:(7 * 3) + 4]
    theta2 = theta[7 * 1:(7 * 1) + 2]
    theta2 = np.reshape(theta2, (theta2.shape[0], 1))
    return theta1, theta2


def predict(theta, x):
    theta1, theta2 = extract_theta(theta)
    a1 = np.insert(x, [0], [1], axis=1).dot(theta1)
    z1 = sigmoid(a1)
    a2 = np.insert(z1, [0], [1], axis=1).dot(theta2)
    h = sigmoid(a2)
    h = np.reshape(h, (h.shape[0], 1))
    return a1, z1, a2, h


def cost_function(theta, *args):
    x, y, lambda_set = args
    m = x.shape[0]
    y = y.reshape((m, 1))
    a1, z1, a2, h = predict(theta, x)
    theta1, theta2 = extract_theta(theta)
    theta1_reg = np.insert(theta1[1:theta1.shape[0], :], [0], [0], axis=0)**2
    theta2_reg = np.insert(theta2[1:theta2.shape[0], :], [0], [0], axis=0)**2
    # print("\t\th values : " + str(h.min()) + " : " + str(h.max()))
    sum_costs = np.sum((-y)*np.nan_to_num(np.log(h)) - (1-y)*np.nan_to_num(np.log(1-h))) + \
                      (lambda_set/2)*(np.sum(theta1_reg) + np.sum(theta2_reg))
    # sum_costs = np.sum((-y)*np.log(h) - (1-y)*np.log(1-h))
    cost_total = (1/m) * sum_costs
    # print("\t\tCost = " + str(cost_total))
    return cost_total


def mini_batch_cost(theta, *args):
    x, y, lambda_set = args
    m = x.shape[0]
    mini_batch_size = 20000
    cost = 0
    for index in range(0, m, mini_batch_size):
        cost = cost + cost_function(theta, *args)
    final_cost = cost/(len(range(0, m, mini_batch_size)))
    return final_cost


def back_prop(theta, *args):
    x, y, lambda_set = args
    m = x.shape[0]
    theta1, theta2 = extract_theta(theta)
    y = y.reshape((m, 1))
    a1, z1, a2, h = predict(theta, x)
    delta2 = h - y
    big_delta2 = np.insert(z1, [0], [1], axis=1).T.dot(delta2)*(1/m)
    delta1 = delta2.dot(theta2.T) * sigmoid_prime(np.insert(z1, [0], [1], axis=1))
    # big_delta1 = a1.T.dot(delta1) + lambda_set * theta1[1:7, :]
    big_delta1 = np.insert(x, [0], [1], axis=1).T.dot(delta1[:, 1:delta1.shape[1]])
    theta1_reg = np.insert(theta1[1:theta1.shape[0], :], [0], [0], axis=0)
    theta2_reg = np.insert(theta2[1:theta2.shape[0], :], [0], [0], axis=0)
    big_delta1 = big_delta1 + lambda_set*theta1_reg
    big_delta2 = big_delta2 + (lambda_set*theta2_reg)*(1/m)
    big_delta1_flat = big_delta1.flatten()
    big_delta1_flat = np.reshape(big_delta1_flat, (big_delta1_flat.shape[0], 1))*(1/m)

    return np.ndarray.flatten(np.concatenate((big_delta1_flat, big_delta2), axis=0))


def mini_batch_back_prop(theta, *args):
    x, y, lambda_set = args
    m = x.shape[0]
    mini_batch_size = 20000
    grad = np.zeros(theta.shape[0])
    for index in range(0, m, mini_batch_size):
        grad = grad + back_prop(theta, *args)
    return grad/(len(range(0, m, mini_batch_size)))


def numerical_gradient(theta, *args):
    epsilon = 0.0001
    gradient = np.zeros(theta.shape)
    theta_epsilon = np.identity(theta.shape[0]) * epsilon
    theta_plus = theta.dot(np.ones((1, theta.shape[0])))
    theta_min = theta.dot(np.ones((1, theta.shape[0])))
    theta_plus = theta_plus + theta_epsilon
    theta_min = theta_min - theta_epsilon
    for index in range(0, theta.shape[0]):
        theta_min_temp = theta_min[:, index]
        theta_plus_temp = theta_plus[:, index]
        gradient[index] = (cost_function(theta_plus_temp, *args) - cost_function(theta_min_temp, *args))/(2*epsilon)

    return np.ndarray.flatten(gradient)


def train_model(theta, *args, max_iteration):
    res1 = optimize.fmin_cg(mini_batch_cost,
                            theta,
                            fprime=mini_batch_back_prop,
                            args=args,
                            maxiter=max_iteration)
    return res1

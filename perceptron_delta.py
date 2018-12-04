#!/usr/bin/env python
"""
This file contains an implementation of a perceptron with sigmoid activation function and delta rule for gradient descend
and a training over the dataset in the dataset.py file.
"""

__author__ = "Giovanni Cavallin, mat. 1206693"
__copyright__ = "Giovanni Cavallin"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

from titanic.dataset import X, y
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def sigmoid(x):
    """
    Numerically-stable sigmoid function. As the PC cannot handle too little/big numbers I "cheated" considering the
    sigmoid(>20) = 1 and sigmoid(<20) = 0
    :param x: the value to evaluate with sigmoid function
    :return: 1 or 0
    """
    if x > 20:
        return 1
    elif x < -20:
        return 0
    return 1 / (1 + np.exp(-x))


def update_w_delta(X, W, delta_W, y_t, learning_rate):
    """
    Update w with the delta gradient descend rule and a sigmoid activation.
    :param X:
    :param W:
    :param delta_W: vector with gradients
    :param y_t: ground truth
    :param learning_rate:
    :return: 1, W + delta_W if the value is not correct yet, 0, W otherwise
    """
    o = sigmoid(np.dot(W.transpose(), X))
    for idx, value in enumerate(X):
        delta_W[idx] = delta_W[idx] + \
                       learning_rate * (y_t - o) * (1 - o) * X[idx]
    if o != y_t:
        return 1, W + delta_W
    else:
        return 0, W


def learn(X, y, learning_rate):
    """
    Learning algorithm. We instantiate W with little casual numbers and delta_W (the vector with the gradients) at 0.
    :param X:
    :param y:
    :param learning_rate:
    :return: the number of iterations, W
    """
    days, attributes = X.shape
    updated = np.ones(days)
    W = np.random.random((attributes, 1))
    delta_W = np.zeros(W.shape)
    iteration = 0
    while updated.any() == 1:
        iteration += 1
        i = np.random.randint(0, days)
        x_temp = np.reshape(X[i], (len(X[i]), 1))
        updated[i], W = update_w_delta(x_temp, W, delta_W, y[i], learning_rate)
    return iteration, W


if __name__ == '__main__':
    learning_rates = np.linspace(0.02, 1, 100)
    iterations = np.zeros(100)
    for idx, learning_rate in enumerate(learning_rates):
        iteration, W = learn(X, y, learning_rate)
        iterations[idx] = iteration
        y_f = np.dot(X, W)
        for idx, value in enumerate(y_f):
            y_f[idx] = sigmoid(y_f[idx])
        acc = f1_score(y, y_f)  # is always 1 since it's a perceptron!
    plt.plot(learning_rates, iterations)
    plt.xlabel("Learning rate")
    plt.ylabel("Iterations")
    plt.show()
    print("W.shape: {w}\nX.transpose.shape(): {x}\ny.shape: {y}".format(w=W.shape, x=X.transpose().shape, y=y.shape))
    # print(iterations)




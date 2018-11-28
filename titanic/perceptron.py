#!/usr/bin/env python
"""
This file contains an implementation of a perceptron and a training over the dataset in the dataset.py file.

As the plot shows it seems that the optimal learning rate over the number of iterations is > 0.04.
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


def real_plot(X, W, y):
    y_n = np.dot(X, W)
    for idx, value in enumerate(y_n):
        y_n[idx] = hard_threshold(value)
    res = y - y_n
    count = 0
    for i in res:
        if i != 0:
            count += 1
    return count

def hard_threshold(x):
    """
    Returns an hard threshold which returns 0 if sign = -1 and 1 otherwise.
    This let the combination of multiple perceptron with uniform input
    :param x: + or - 1
    :return: 1 for positive, 0 for negative
    """
    return 0 if np.sign(x) <= 0 else 1


def update_w(x, w, t, learning_rate):
    o = hard_threshold(np.dot(w.transpose(), x))
    if o != t:
        x_m = learning_rate * (t - o) * x
        return 1, w + x_m
    else:
        return 0, w
    # return (1, w + x_m) if o != t else (0, w)

# def set_initial_w(shape=(1, 1)):
#     return np.random.rand(shape)


def learn(X, y, learning_rate):
    values, features = X.shape
    updated = np.ones(values)
    W = np.random.random((features, 1))
    # print("W before training: \n{}".format(W))
    iteration = 0
    y_ax = np.linspace(0, len(y), len(y))
    while updated.any() == 1:
        iteration += 1
        i = np.random.randint(0, values)
        x_temp = np.reshape(X[i], (len(X[i]), 1))
        updated[i], W = update_w(x_temp, W, y[i], learning_rate)
        # plt.plot(1, real_plot(X, W, y))
        # plt.pause(0.05)
    # plt.show()
    # print("W after {it} iteration(s): \n{w}".format(it=iteration, w=W))
    return iteration, W


if __name__ == '__main__':
    iteration, W = learn(X, y, 0.00004)
    # learning_rates = np.linspace(0.001, 0.035, 3)
    # iterations = []
    # for learning_rate in learning_rates:
    #     iteration, W = learn(X, y, learning_rate)
    #     iterations.append(iteration)
    # plt.plot(learning_rates, iterations)
    # plt.xlabel("Learning rate")
    # plt.ylabel("Iterations")
    # plt.show()
    print("W.shape: {w}\nX.transpose.shape(): {x}\ny.shape: {y}".format(w=W.shape, x=X.transpose().shape, y=y.shape))
    # print(iterations)




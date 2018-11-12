#!/usr/bin/env python
"""
"""

__author__ = "Giovanni Cavallin, mat. 1206693"
__copyright__ = "Giovanni Cavallin"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

# from dataset import X, y
import numpy as np
import matplotlib.pyplot as plt


# def hard_threshold(x):
#     """
#     Returns an hard threshold which returns 0 if sign = -1 and 1 otherwise.
#     This let the combination of multiple perceptron with uniform input
#     :param x: + or - 1
#     :return: 1 for positive, 0 for negative
#     """
#     return 0 if np.sign(x) < 0 else 1


# def update_w(x, w, t, learning_rate):
#     o = hard_threshold(np.dot(w, x))  # because in my dataset: t in (0, 1)
#     x_m = np.transpose(learning_rate * (t - o) * x)
#     return (1, w + x_m) if o != t else (0, w)

# def set_initial_w(shape=(1, 1)):
#     return np.random.rand(shape)

def sigmoid(y, derivative=False):
    sigm = np.divide(1., (1. + np.exp(-y)))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def update_delta_w_linear(x, y, w, delta_w, learning_rate):
    o = np.dot(w, x)
    u = learning_rate*(y-o)*x
    if np.linalg.norm(u, 4) < 10**-5:
        return False, delta_w
    else:
        delta_w = delta_w + u
        return True, delta_w


def update_delta_w_sigmoid(x, y, w, delta_w, learning_rate):
    o = np.dot(w, x)
    delta_w = delta_w + learning_rate*(y - o)*sigmoid(o)*(1 - sigmoid(o))*x
    return delta_w


def learn(X, y, learning_rate):
    days, attributes = X.shape
    # updated = np.ones(days)
    toll = 10**-5
    step = toll + 10
    W = np.random.rand(X.shape[0], X.shape[1])
    delta_W = np.zeros(W.shape)
    # print("W before training: \n{}".format(W))
    iteration = 0
    update = True
    while update:
        iteration += 1
        day = np.random.randint(0, days)
        update, delta_W[day] = update_delta_w_linear(X[day], y[day], W[day], delta_W[day], learning_rate)
        # step = np.linalg.norm(delta_W[day])
        W[day] = W[day] + delta_W[day]
    # print("W after {it} iteration(s): \n{w}".format(it=iteration, w=W))
    return iteration, W



def f(x):
    """function to approximate by polynomial interpolation"""
    return x * np.sin(x)


if __name__ == '__main__':
    learning_rates = np.linspace(0.0001, 0.002, 1000)
    iterations = []


    # generate points used to plot, 100 numbers from 0 to 10
    x_plot = np.linspace(0, 10, 100)

    # generate points and keep a subset of them
    X = np.linspace(0, 10, 100)
    rng = np.random.RandomState(0)
    rng.shuffle(X)
    X = np.sort(X[:20])
    y = f(X)

    # create matrix version of these arrays
    X = X[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold', 'blue']
    lw = 3
    plt.plot(
        x_plot,
        f(x_plot),
        color='cornflowerblue',
        linewidth=lw,
        label='ground truth'
    )
    plt.scatter(
        X,
        y,
        color='navy',
        s=30,
        marker='o',
        label='training points'
    )
    for learning_rate in learning_rates:
        iteration, W = learn(X, y, learning_rate)
        plt.plot(
            X,
            X + W,
            # color=colors[count],
            # linewidth=lw,
            # label='degree {}'.format(degree)
        )
    # plt.plot(learning_rates, iterations)
    # plt.xlabel("Learning rate")
    # plt.ylabel("Iterations")
    plt.show()
    print("W.shape: {w}\nX.transpose.shape(): {x}\ny.shape: {y}".format(w=W.shape, x=X.transpose().shape, y=y.shape))
    print("W: {}".format(W))




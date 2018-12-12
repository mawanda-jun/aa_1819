#!/usr/bin/env python
"""
This files contains the exercise that has been proposed for lesson 14.
We had to demonstrate that dividing the training set in little pieces and using high values for
p, the polynomial grade, we should have been able to get a lower value for variance than the one obtained with
the training with the full set.
"""

__author__ = "Giovanni Cavallin, mat. 1206693"
__copyright__ = "Giovanni Cavallin"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def residual_variance(y_true, y_pred):
    """
    Manual calculation of residual variance. It is obtained calculating the square error for every entry of every point,
    summarizing all of them and then dividing the sum for n-2, where n is the length of the array.
    Please refer to this specification: https://www.coursera.org/lecture/regression-models/residual-variance-WMAET
    :param y_true: ground truth
    :param y_pred: the estimated y
    :return: float, residual variance of our model
    """
    sum = 0
    for idx, value in enumerate(y_true):
        sum += (value - y_pred[idx]) ** 2
    return float(sum / len(y_true))


def explained_variance(y_true, y_pred):
    y_med = np.average(y_true)
    sum = 0
    for idx, value in enumerate(y_pred):
        sum += (value-y_med)**2
    return float(sum / len(y_pred))


def f(x):
    """
    function to approximate by polynomial interpolation
    :return:
    """
    return 1 + 2*np.sqrt(x) + 20*x + x**(3/2) - x**2


if __name__ == '__main__':
    # variable to tune the model
    DIM_TRAINING = 1000000  # 100000
    DIM_VALIDATION = 1000  # 100
    X_LENGTH = 80  # 40

    degree = 8  # 8
    k = 3000  # 100, is the number of the examples in the mini-training set

    coef_dim = degree+1  # dim of coefficients to take into account when averaging the models
    tr_meno_va = int((DIM_TRAINING - DIM_VALIDATION) / k)  # dimension of the actual training set

    # create the dataset
    x = np.linspace(0, X_LENGTH, DIM_TRAINING)

    # randomize it
    rng = np.random.RandomState(0)
    rng.shuffle(x)

    # extract the part for the validation
    X_va = x[:DIM_VALIDATION][:, np.newaxis]

    # deleting it from training set
    x = np.delete(x, X_va)

    # sort validation set
    X_va = np.sort(X_va, axis=0)
    y_va = f(X_va)

    # prepare mini-datasets to be loaded after
    X = np.zeros((tr_meno_va, k))
    y = np.zeros((tr_meno_va, k))

    for idx in range(tr_meno_va):
        X[idx] = np.sort(x[idx * k:(idx + 1) * k])
        y[idx] = f(X[idx])

    # try with full dataset on single validation set
    X_full = np.resize(X, (DIM_TRAINING, 1))
    y_t_full = f(X_full)

    # make prediction with full training set
    poly = PolynomialFeatures(degree)
    X_full_ = poly.fit_transform(X_full)
    X_t_ = poly.fit_transform(X_va)
    lg = LinearRegression()
    lg.fit(X_full_, y_t_full)
    y_pred = lg.predict(X=X_t_)

    print("variance with full training set: {v}".format(v=residual_variance(y_va, y_pred)))

    # make prediction with small dataset and averaging models
    # score = np.zeros(tr_meno_va)
    # var = np.zeros(tr_meno_va)

    # to average models I take into account the coefficients and the intercepts of every model to make an average later.
    coefs = np.zeros((tr_meno_va, coef_dim))
    intercepts = np.zeros((tr_meno_va, 1))
    poly = PolynomialFeatures(degree=degree)
    X_va_ = poly.fit_transform(X_va)
    for idx in range(tr_meno_va):
        y_ = y[idx]
        p = X[idx]
        X_ = poly.fit_transform(X[idx][:, np.newaxis])
        lg = LinearRegression()
        lg.fit(X_, y_)
        # y_pred = lg.predict(X_va_)
        # score[idx] = lg.score(X_va_, y_va)
        # var[idx] = residual_variance(y_va, y_pred)
        coefs[idx] = lg.coef_
        intercepts[idx] = lg.intercept_
    # print("min var: {}".format(np.min(var)))

    # now I average the coefficients and the intercepts of every model
    coef = np.zeros((coef_dim, 1))
    coefs = coefs.T
    for idx, value in enumerate(coefs):
        coef[idx] = np.average(value)
    intercept = np.mean(intercepts)

    # make another regressor to see if the means performs better
    best_lg = LinearRegression()
    best_lg.coef_ = np.resize(coef, (len(coef),))
    best_lg.intercept_ = intercept

    # now I make the inference with the new "mean" linear model
    y_best_pred = best_lg.predict(X=X_t_)
    best_score = best_lg.score(X_t_, y_va)
    best_var = residual_variance(y_va, y_best_pred)
    print("variance with mean model: {bv}".format(bv=best_var))

    # plot results to see what is happening
    lw = 2
    plt.plot(X_va, f(X_va), color='cornflowerblue', linewidth=lw+2,
             label="ground truth")
    plt.scatter(X_va, f(X_va), color='navy', s=30, marker='o', label="Validation points")
    plt.plot(X_va, y_pred, color='teal', linewidth=lw, label='full training')
    plt.plot(X_va, y_best_pred, color='gold', linewidth=lw, label='mean training')

    plt.legend(loc='lower left')
    plt.show()

#!/usr/bin/env python
"""
This file contains the implementations of binary operation through perceptrons with some use-case at the bottom.

Every function has been implemented trying to be as general as possible. This is why I always use the implementation of
a matrix and the np.dot function. Something that can be appreciated: I treat the x_0 and w_0 values separately and then
merge them into the original "data set", so values with different meanings (as they are data and bias) can be stored
separately.

This kind of function let the programmer create an easy network of perceptrons by combining them together.
"""

__author__ = "Giovanni Cavallin, mat. 1206693"
__copyright__ = "Giovanni Cavallin"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

import numpy as np


def hard_threshold(x):
    """
    Returns an hard threshold which returns 0 if sign = -1 and 1 otherwise.
    This let the combination of multiple perceptron with uniform input
    :param x: + or - 1
    :return: 1 for positive, 0 for negative
    """
    return 0 if np.sign(x) < 0 else 1


def percep_or(A, B):
    """
    Returns A OR B with a boolean perceptron
    :param A: boolean value
    :param B: boolean value
    :return: 1 if true, -1 if false
    """
    x = np.empty((2, 1))
    w = np.empty((1, 2))
    x[0][0] = A
    x[1][0] = B
    w.fill(1)
    x_0 = [[1]]
    w_0 = [[-0.5]]
    x = np.insert(x, 0, x_0)
    w = np.insert(w, 0, w_0)
    H = hard_threshold(np.dot(w, x))
    return H


def percep_and(A, B):
    """
    Returns A AND B with boolean perceptron
    :param A: boolean value
    :param B: boolean value
    :return: 1 if true, -1 if false
    """
    x = np.empty((2, 1))
    w = np.empty((1, 2))

    x[0][0] = A
    x[1][0] = B

    w.fill(1)
    x_0 = np.ones((1, 1))
    w_0 = np.empty((1, 1))
    w_0[0][0] = - np.shape(x)[0] + 0.5
    x = np.insert(x, 0, x_0)
    w = np.insert(w, 0, w_0)
    H = hard_threshold(np.dot(w, x))
    return H


def percep_not(X):
    """
    Returns NOT X with boolean perceptron
    :param X: boolean value
    :return: 1 if true, -1 if false
    """
    x = np.ones((1, 1))
    w = np.empty((1, 1))
    x[0][0] = X
    w[0][0] = -2
    x_0 = np.ones((1, 1))
    w_0 = np.ones((1, 1))
    x = np.insert(x, 0, x_0)
    w = np.insert(w, 0, w_0)
    H = hard_threshold(np.dot(w, x))
    return H


if __name__ == '__main__':
    # A AND (NOT B):
    # A | B | NOT B | A AND (NOT B)
    # 0 | 0 |   1   |  0
    # 1 | 0 |   1   |  1
    # 0 | 1 |   0   |  0
    # 1 | 1 |   0   |  0

    A = 1
    B = 0
    print("A: {a}\nB: {b}\n".format(a=A, b=B))
    print("The seven basic logic gates:")
    print("AND: {}".format(percep_and(A, B)))
    print("OR: {}".format(percep_or(A, B)))
    print("XOR (aka: (A AND (NOT B)) OR ((NOT A) AND B)): {}".format(
        percep_or(percep_and(A, percep_not(B)), percep_and(B, percep_not(A)))
    ))
    print("NOT (A): {}".format(percep_not(A)))
    print("NAND: {}".format(percep_not(percep_and(A, B))))
    print("NOR: {}".format(percep_not(percep_or(A, B))))
    print("XNOR: {}".format(
        percep_not(percep_or(percep_and(A, percep_not(B)), percep_and(B, percep_not(A))))
    ))
    print("\nSome credits:")
    print("A AND (NOT B): {}".format(percep_and(A, percep_not(B))))

    # XOR: (A AND (NOT B)) OR ((NOT A) AND B)
    # A | B | NOT A | NOT B | XOR
    # 0 | 0 |   1   |   1   |  0
    # 1 | 0 |   0   |   1   |  1
    # 0 | 1 |   1   |   0   |  1
    # 1 | 1 |   0   |   0   |  0







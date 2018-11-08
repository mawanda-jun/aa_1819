#!/usr/bin/env python
"""
This file contains the dataset of the book for find-s and candidate-elimination algorithms.

In particular I tried to make it as readable as possible while maintaining performances. So even if the database is
built in a "human" style - with names and references - the machine only sees 0 and 1.

Two functions to go from number to names and vice versa are provided.
"""

__author__ = "Giovanni Cavallin, mat. 1206693"
__copyright__ = "Giovanni Cavallin"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

import numpy as np

positive = 1  # X_true_value
negative = 0  # X_false_value
anyone = -1  # X_any_value
noone = -2  # X_no_value

labels = {
    "sky": ('sunny', 'rainy'),
    "air_temp": ('warm', 'cold'),
    "humidity": ('normal', 'high'),
    "wind": ('strong', 'weak'),
    "water": ('warm', 'cool'),
    "forecast": ('same', 'change'),
    "enjoy_sport": ('yes', 'no')
}
# print(labels)

order = [
    "sky",
    "air_temp",
    "humidity",
    "wind",
    "water",
    "forecast",
    "enjoy_sport"
]  # not needed for python > 3.6 as dict will be ordered as inserted


def string_to_num(name):
    for nm in order:
        if name in labels[nm][0]:
            return positive
        if name in labels[nm][1]:
            return negative
    if name == "any":
        return anyone
    if name == "noone":
        return noone


def num_to_string(attr_num, value):
    return labels[order[attr_num]][value]


def print_hyp(hyp):
    a = "<"
    b = ""
    for i in range(len(hyp)):
        if hyp[i] == noone:
            b = "0"
        elif hyp[i] == anyone:
            b = "?"
        elif hyp[i] != anyone:
            b = num_to_string(attr_num=i, value=0)
        a += b + ", "
    a += ">"

    print('Condition(s) to be satisfied:\n{}'.format(a))

X = np.array([
    [string_to_num(labels[order[0]][0]), string_to_num(labels[order[1]][0]), string_to_num(labels[order[2]][0]), string_to_num(labels[order[3]][0]), string_to_num(labels[order[4]][0]), string_to_num(labels[order[5]][0])],
    [string_to_num(labels[order[0]][0]), string_to_num(labels[order[1]][0]), string_to_num(labels[order[2]][1]), string_to_num(labels[order[3]][0]), string_to_num(labels[order[4]][0]), string_to_num(labels[order[5]][0])],
    [string_to_num(labels[order[0]][1]), string_to_num(labels[order[1]][1]), string_to_num(labels[order[2]][1]), string_to_num(labels[order[3]][0]), string_to_num(labels[order[4]][0]), string_to_num(labels[order[5]][1])],
    [string_to_num(labels[order[0]][0]), string_to_num(labels[order[1]][0]), string_to_num(labels[order[2]][1]), string_to_num(labels[order[3]][0]), string_to_num(labels[order[4]][1]), string_to_num(labels[order[5]][1])],
])
y = np.array([
    [string_to_num(labels[order[6]][0])],
    [string_to_num(labels[order[6]][0])],
    [string_to_num(labels[order[6]][1])],
    [string_to_num(labels[order[6]][0])],
])

# print(X)

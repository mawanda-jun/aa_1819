#!/usr/bin/env python
"""
This file contains an implementation of a perceptron and a training over the dataset in the dataset.py file.

As the plot shows it seems that the optimal learning rate over the number of iterations is > 0.04.
"""

__author__ = "Giovanni Cavallin, mat. 1206693, Alberto Bezzon, Alberto Gallinaro"
__copyright__ = "Giovanni Cavallin, mat. 1206693, Alberto Bezzon, Alberto Gallinaro"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Giovanni Cavallin"
__email__ = "giovanni.cavallin.1@studenti.unipd.it"
__status__ = "Production"

import sklearn.preprocessing as sk_prep
import pandas as pd
import os
import numpy as np

passenger_cat = {
    "1": 0,
    "2": 1,
    "3": 2
}
num_passenger_cat = 3
sex_cat = {
    "male": 0,
    "female": 1
}
embarked_cat = {
    "C": 0,  # Cherbourg
    "Q": 1,  # Queenstown
    "S": 2  # Southampton
}


def average(arr):
    support = np.zeros((len(arr)))
    i = 0
    for idx, value in enumerate(arr):
        if 'nan' not in str(value):
            support[i] = value
            i += 1
    support = support[1:i]
    return np.average(support)


def one_hot(cls, classes=1):
    """
    Method which create a category and returns an array with all the values transformed
    :param cls: origin array with categorized elements
    :param classes: number of classes in which we want our array to be encoded
    :return: encoded array of dimension (classes, len(cls))
    """
    v = [[0]]
    idx = 1
    while idx < classes:
        v += [[idx]]
        idx += 1
    enc = sk_prep.OneHotEncoder(categories='auto')
    enc.fit(v)
    result = np.zeros((len(cls), classes))
    for idx, value in enumerate(cls):
        result[idx] = enc.transform([[value]]).toarray()
    return result


def class_category(cls):
    """
    Since classes are already classified we make it start from 0
    :param cls:
    :return:
    """
    return cls - 1


def sex_category(s):
    """
    Male is 0, female is 1
    :param s:
    :return:
    """
    return sex_cat[s]


def age_category(age, average=-1):
    """
    We divided the age into 5 steps taking care of the responsibility that a person has of himself
    and looking at how mush "strength" it has. This is an insteresting parameter to change while
    seeing inference.
    In case of nan values we put the average eta of the people who was embarked
    :param age: int with number
    :param average: average eta of people
    :return: class. -1 is returned in case of some strange value
    """
    if 'nan' in str(age):
        age = average
    if age < 11:
        return 0
    elif age < 20:
        return 1
    elif age < 40:
        return 2
    elif age < 60:
        return 3
    elif age < 200:
        return 4
    return -1


def fare_category(fare, average = -1):
    """
    We divided the fare into 4 sections which correspond to the classes, almost. We look at some historical
    site in which the costs were reported and we followed it looking at the dataset
    :param fare: float, cost of
    :return:
    """
    if 'nan' in str(fare):
        fare = average
    if fare < 10:
        return 0
    elif fare < 30:
        return 1
    elif fare < 70:
        return 2
    elif fare < 1000:
        return 3
    return -1


def ticket_category(ticket):
    """
    To implement eventually
    :param ticket:
    :return:
    """
    if "STON" in ticket:
        return 0
    elif "C.A." in ticket or "CA" in ticket:
        return 1
    elif "SOTON" in ticket:
        return 2
    elif "F.C." in ticket:
        return 3
    elif "A/" in ticket:
        return 4
    elif "SC" in ticket:
        return 5
    elif "PC" in ticket:
        return 6
    elif "S.O." in ticket:
        return 7
    return 8


def cabin_category(cab):
    """
    To implement eventually
    :param cab:
    :return:
    """
    cab = str(cab)
    if "A" in cab:
        return 0
    elif "B" in cab:
        return 1
    elif "C" in cab:
        return 2
    elif "D" in cab:
        return 3
    elif "E" in cab:
        return 4
    elif "D" in cab:
        return 5
    elif "F" in cab:
        return 6
    elif "G" in cab:
        return 7
    return -1


def embarked_category(emb):
    """
    To implement eventually
    :param emb:
    :return:
    """
    if emb == "C" or emb == "Q" or emb == "S":
        return embarked_cat[emb]
    else:
        return -1



# get the current path
path = "titanic"

# Passenger 0, nessuna correzione
# Survived 1, è la nostra y
# Pclass 2, 3 classi, codifica OneHot
# Name 3, si elimina
# Sex 4, classi, codifica OneHot (modifica in 0/1)
# Age 5, cluster
# SibSp 6, togliamo perché: abbiamo ritenuto che non fosse influente perché avere un fratello non
# comporterebbe una maggiore probabilit' di salvarsi
# Parch 7, normalizzazione, centramento. Non prendiamo in considerazione la standardizzazione perchè abbiamo
# non abbiamo notato particolari corrispondenze tra il numero dei figli e la probabilità di salvarsi,
# soprattutto dato il sesso
# Ticket 8, togliamo
# Fare 9, teniamo, clusterizzare
# Cabin 10, togliamo come visto a lezione
# Embarked 11, lo teniamo per dopo.

# per il test set:
# PassengerId 0,
# Pclass 1,
# Name 2,
# Sex 3,
# Age 4,
# SibSp 5,
# Parch 6,
# Ticket 7,
# Fare 8,
# Cabin 9,
# Embarked 10
# create path/to/dataset
train_path = os.path.join(path, "train.csv")
test_path = os.path.join(path, "test.csv")
res_test_path = os.path.join(path, "gender_submission.csv")

# create pandas object with dataset
train_set = pd.read_csv(train_path)
test_set = pd.read_csv(test_path)
res_test_set = pd.read_csv(res_test_path)

# elaboration of dataset
# Pandas DataFrame to Numpy. Then transpose it to have only a feature for each column
T = train_set.values.transpose()
T_t = test_set.values.transpose()

# PASSENGER CLASSES
# train
for idx, value in enumerate(T[2]):
    T[2][idx] = class_category(value)
train_classes = one_hot(T[2], num_passenger_cat)
# test
for idx, value in enumerate(T_t[1]):
    T_t[1][idx] = class_category(value)
test_classes = one_hot(T_t[1], num_passenger_cat)

# SEX CLASSES
# train
for idx, value in enumerate(T[4]):
    T[4][idx] = sex_category(value)
train_sexes = one_hot(T[4], 2)
# test
for idx, value in enumerate(T_t[3]):
    T_t[3][idx] = sex_category(value)
test_sexes = one_hot(T_t[3], 2)

# AGE CLASSES
# train
# we want to substitute the average age of the known people to the unkwown. So we need to prune all
# the nan values in order to retrieve the average and calculate it
for idx, value in enumerate(T[5]):  # Age
    T[5][idx] = age_category(value, average(T[5]))

train_ages = one_hot(T[5], 5)

# test
for idx, value in enumerate(T_t[4]):  # Age
    T_t[4][idx] = age_category(value, average(T_t[4]))

test_ages = one_hot(T_t[4], 5)

# PARENT/CHILDREN CLASSES
# train
train_parch = [T[7]]
normalizer = sk_prep.Normalizer()
normalizer.fit(train_parch)
train_parch = normalizer.transform(train_parch)
# test
test_parch = [T_t[6]]
normalizer = sk_prep.Normalizer()
normalizer.fit(test_parch)
test_parch = normalizer.transform(test_parch)

# FARE CLASSES
# train
for idx, value in enumerate(T[9]):  # Fare
    T[9][idx] = fare_category(value)
train_fares = one_hot(T[9], 4)
# test
for idx, value in enumerate(T_t[8]):  # Fare
    T_t[8][idx] = fare_category(value, average(T_t[8]))
test_fares = one_hot(T_t[8], 4)

# # Now we take the y value from dataset
y = T[1]

# Now we create our dataset
X = np.concatenate((train_classes.T, train_ages.T, train_sexes.T, train_parch, train_fares.T)).T
X_t = np.concatenate((test_classes.T, test_ages.T, test_sexes.T, test_parch, test_fares.T)).T

y_t = res_test_set.values
y_t = y_t.T
y_t = y_t[1]
y_t = y_t.T

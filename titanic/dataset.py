import pandas as pd
import os
import numpy as np
import sklearn.preprocessing as sk_prep

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


def fare_category(fare):
    """
    We divided the fare into 4 sections which correspond to the classes, almost. We look at some historical
    site in which the costs were reported and we followed it looking at the dataset
    :param fare: float, cost of
    :return:
    """
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


if __name__ == '__main__':
    # get the current path
    path = os.getcwd()

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

    # create path/to/dataset
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    res_test_path = os.path.join('gender_submission.csv')

    # create pandas object with dataset
    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    res_test_set = pd.read_csv(res_test_path, usecols=[1])

    # elaboration of dataset
    # Pandas DataFrame to Numpy. Then transpose it to have only a feature for each column
    X = train_set.values.transpose()

    # passenger classes
    for idx, value in enumerate(X[2]):
        X[2][idx] = class_category(value)
    classes = one_hot(X[2], num_passenger_cat)

    # sex classes
    for idx, value in enumerate(X[4]):
        X[4][idx] = sex_category(value)
    sexes = one_hot(X[4], 2)

    # age classes
    # we want to substitute the average age of the known people to the unkwown. So we need to prune all
    # the nan values in order to retrieve the average and calculate it
    support = np.zeros((len(X[5])))
    i = 0
    for idx, value in enumerate(X[5]):
        if 'nan' not in str(value):
            support[i] = value
            i += 1
    support = support[1:i]
    average = np.average(support)

    for idx, value in enumerate(X[5]):  # Age
        X[5][idx] = age_category(value, average)

    ages = one_hot(X[5], 5)

    # parent/children classes 7
    parch = [X[7]]
    normalizer = sk_prep.Normalizer()
    normalizer.fit(parch)
    parch = normalizer.transform(parch)

    # fare classes
    for idx, value in enumerate(X[9]):  # Fare
        X[9][idx] = fare_category(value)
    fares = one_hot(X[9], 4)

    # # Now delete the y value from dataset so we can separate the features from the result
    y = np.array(X[1])
    y = np.resize(y, (len(X[1]), 1))

    # Now we create our dataset
    X = np.concatenate((classes.T, ages.T, sexes.T, parch, fares.T)).T

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

from sklearn.svm import SVR
from sklearn.metrics import f1_score
from titanic.dataset import X, y, X_t, y_t
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread
import warnings
warnings.filterwarnings('ignore')


def hard_threshold(x):
    """
    Returns an hard threshold which returns 0 if sign = -1 and 1 otherwise.
    This let the combination of multiple perceptron with uniform input
    :param x: + or - 1
    :return: 1 for positive, 0 for negative
    """
    return 0 if x <= 0.5 else 1


def infere(X, y, X_t, y_t, c, k, e, d=3):
    clf = SVR(
        kernel=k,
        gamma='auto',
        degree=d,
        C=c,
        epsilon=e,
        verbose=False,
        cache_size=1000
    )
    clf.fit(X, y)
    y_f = clf.predict(X_t)
    y_f = y_f.T
    for idx, value in enumerate(y_f):
        y_f[idx] = hard_threshold(value)

    return f1_score(y_t, y_f, labels=np.unique(y_t))


class MyThread(Thread):
    def __init__(self, name, parameters, kernel):
        Thread.__init__(self)
        self.name = name
        self.parameters = parameters
        self.kernel = kernel
        self.accuracy = None

    def run(self):
        acc = np.zeros((len(parameters['C']) * len(parameters['epsilon']), 3))
        # try for rbf kernel
        i = 0
        for c in parameters['C']:
            for e in parameters['epsilon']:
                acc[i] = [c, e, infere(X, y, X_t, y_t, c, self.kernel, e)]
                i += 1
        self.accuracy = acc
        print('Accuracy with {k}:\n{v}\n'.format(k=self.kernel, v=find_max(acc)))


def find_max(arr):
    max = []
    for val in arr:
        if max == []:
            max = val
        else:
            # teniamo un margine del 10% sul risultato migliore per vedere se possiamo scendere leggermente di
            # C ed epsilon per evitare overfitting

            if val[2] >= max[2]:
                # puntiamo a minimizzare anche epsilon e C. Dal momento che sono ordinati, verrà
                # fuori quello con C/epsilon più piccoli
                # if val[1] < max[1]:
                    # piu' piccolo il C meglio e'!
                    # if val[0] < max[0]:
                max = val
    return max


if __name__ == '__main__':
    # generate a distribution around C = 1 (sklearn default for rbf) to see how accuracy varies
    C = np.linspace(0.5, 0.01, 100)
    # generate a distribution around epsilon = 0.1 (sklearn default for rbf) to see how accuracy varies
    epsilon = np.linspace(0.01, 0.001, 100)

    parameters = {
        "C": C,
        "epsilon": epsilon
    }

    threads = [MyThread('Thread1', parameters, 'rbf'), MyThread('Thread2', parameters, 'poly'), MyThread('Thread3', parameters, 'sigmoid')]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    colors = ['red', 'blue', 'green']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    for idx, thread in enumerate(threads):
        val = thread.accuracy.T
        ax.scatter(val[0], val[1], val[2], zdir='z', c=colors[idx], marker=',')
    ax.set_xlabel('C')
    ax.set_ylabel('epsilon')
    ax.set_zlabel('F1')
    plt.savefig("demo1.png")
    plt.show()


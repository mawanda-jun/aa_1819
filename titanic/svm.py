import sklearn.svm as svm
import numpy
from dataset import X, y

clf = svm.SVR()
print(clf.fit(X, y))

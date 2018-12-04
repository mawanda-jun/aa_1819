import sklearn.svm as svm
from titanic.dataset import X, y, X_t, y_t
import matplotlib.pyplot as ply

clf = svm.SVR()
clf.fit(X, y)
y_f = clf.predict(X_t)

ply.plot(y_f)
ply.show()

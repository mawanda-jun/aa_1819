import numpy as np
from dataset import X, y, positive, negative, anyone, noone, print_hyp

days, attributes = X.shape

print("Days: {d}\nAttributes: {a}".format(d=days, a=attributes))
print("dataset:\nX: {x}\ny:{Y}".format(x=X, Y=y))

# initialize first hyp value
hyp = np.empty(attributes)
hyp.fill(noone)  # most specific value

print("hyp:",  hyp)
i = 0
positive_found = False
while i < attributes and not positive_found:
    if y[i] == positive:
        hyp = X[i]  # first positive condition in X
        positive_found = True

for day in range(days)[i:]:
    print("X:\t", X[day])
    if y[day][0] == positive:
        print("taken!")
        for attribute in range(attributes):
            # if X[day][attribute] != negative and hyp[attribute] == X[day][attribute]:
            if X[day][attribute] != hyp[attribute]:
                hyp[attribute] = anyone
            # else:
            #     hyp[attribute] = negative
    else:
        print("not taken!")
    print("hyp:", hyp)

print_hyp(hyp)

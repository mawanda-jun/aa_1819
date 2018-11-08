import numpy as np

n = 10  # dimension of array
x = np.empty((n, 1))
w = np.empty((1, n))

# OR implementation with perceptron

x.fill(1)
w.fill(1)
# x[0][0] = 0

# conditions on first element ("b")
# x[0][0] = 1
x_0 = [[1]]
# print("x_0 shape: {}".format(np.shape(x_0)))
# w[0][0] = -0.5
w_0 = [[-0.5]]

x = np.append(x, x_0)
w = np.append(w, w_0)
H = np.sign(np.dot(w, x))
print("X: {}".format(x))
print("w: {}".format(w))
print("OR between elements: {}".format(H))


# -------------------------------------------

# AND implementation with perceptron
x = np.empty((n, 1))
w = np.empty((1, n))

x.fill(1)
w.fill(1)
x[0][0] = 0  # makes condition false

# conditions on first element ("b")
# x[0][0] = 1
x_0 = np.ones((1, 1))
w_0 = np.empty((1, 1))
w_0[0][0] = - np.shape(x)[0] + 0.5

# w[0][0] = - np.shape(x)[0] + 1 + 0.5  # because x-shape is n+1
x = np.append(x, x_0)
w = np.append(w, w_0)

H = np.sign(np.dot(w, x))
print("X: {}".format(x))
print("w: {}".format(w))
print("AND between elements: {}".format(H))

# ------------------------------------------

# NOT implementation with perceptron
x = np.ones((1, 1))
w = np.empty((1, 1))

w[0][0] = -2

# conditions on "b"
x_0 = np.ones((1, 1))
w_0 = np.ones((1, 1))

x = np.append(x, x_0)
w = np.append(w, w_0)

H = np.sign(np.dot(w, x))
print("X: {}".format(x))
print("w: {}".format(w))
print("NOT between elements: {}".format(H))


# ------------------------------------------------

# A AND (NOT B):
# A | B | NOT B | A AND (NOT B)
# 0 | 0 |   1   |  0
# 1 | 0 |   1   |  1
# 0 | 1 |   0   |  0
# 1 | 1 |   0   |  0

x = np.empty((2, 1))
w = np.empty((1, 2))

A = x[0][0] = 1  # A
B = x[1][0] = 0  # B

# first we have to make not of B. First perceptron
# we treat only B input
w_B = w[0][1] = -2
B_0 = np.ones((1, 1))
w_B_0 = np.ones((1, 1))
B_1 = np.append(B, B_0)
w_1 = np.append(w_B, w_B_0)
not_B = np.sign(np.dot(B_1, w_1))
print("A: {a}\nB: {b}".format(a=A, b=B))
print("NOT B: {}".format(not_B))

# then we have to make A AND NOT B
# rewrite x and w with new values
x = np.empty((2, 1))
x[0][0] = A
x[1][0] = not_B
w.fill(1)

x_0 = np.ones((1, 1))
w_0 = np.empty((1, 1))
w_0[0][0] = - np.shape(x)[0] + 0.5

# w[0][0] = - np.shape(x)[0] + 1 + 0.5  # because x-shape is n+1
x = np.append(x, x_0)
w = np.append(w, w_0)

H = np.sign(np.dot(w, x))
print("A AND (NOT B)): {}".format(H))

# ----------------------------------------------------

# XOR: (A AND (NOT B)) OR ((NOT A) AND B)
# A | B | NOT A | NOT B | XOR
# 0 | 0 |   1   |   1   |  0
# 1 | 0 |   0   |   1   |  1
# 0 | 1 |   1   |   0   |  1
# 1 | 1 |   0   |   0   |  0

# much simpler with functions!

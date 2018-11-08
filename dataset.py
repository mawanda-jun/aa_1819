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
    if name in (labels[order[0]][0], labels[order[1]][0], labels[order[2]][0], labels[order[3]][0], labels[order[4]][0], labels[order[5]][0], labels[order[6]][0]):
        return positive
    if name in (labels[order[0]][1], labels[order[1]][1], labels[order[2]][1], labels[order[3]][1], labels[order[4]][1], labels[order[5]][1], labels[order[6]][1]):
        return negative
    if name == "any":
        return anyone
    if name == "noone":
        return noone


def num_to_string(attr_num, value):
    return labels[order[attr_num]][value]

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

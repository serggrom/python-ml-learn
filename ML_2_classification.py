
from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_iris


def is_virginica_test(fi, t, reverse, example):
    "Apply threshold model to a new example"
    test = example[fi] > t
    if reverse:
        test = not test
    return test

data = load_iris()

features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names


for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'o'
    elif t == 2:
        c = 'b'
        marker == 'x'
    plt.scatter(features[target == t, 0], features[target == t, 1],
                marker=marker, c=c)
    plt.show()


'''
fig, axes = plt.subplots(2, 3)

pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

color_markers = [
    ('r', '>'),
    ('g', 'o'),
    ('b', 'x')
]



for i, (p0, p1) in enumerate(pairs):
    ax = axes.flat[i]

    for t in range(3):
        c, marker = color_markers[t]
        ax.scatter(
            features[target == t, p0], features[target == t, p1],
            marker=marker, c=c
        )
        ax.set_xlabel(feature_names[p0])
        ax.set_ylabel(feature_names[p1])
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
fig.savefig('figure1.jpg')
'''


labels = target_names[target]

plength = features[:, 2]

is_setosa = (labels == 'setosa')

max_setosa = plength[is_setosa].max()

min_non_setosa = plength[~is_setosa].min()
print("Maximum of setosa: {0}.".format(max_setosa))

print("Minimum of setosa: {0}.".format(min_non_setosa))

features = features[~is_setosa]
labels = labels[~is_setosa]

is_virginica = (labels == 'virginica')

best_acc = -1.0

for fi in range(features.shape[1]):
    thresh = features[:, fi]
    for t in thresh:
        feature_i = features[:, fi]
        pred = (feature_i > t)
        acc = (pred == is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc > acc:
            reverse = True
            acc = rev_acc
        else:
            reverse = False
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse

print(best_acc, best_fi, best_t, best_reverse)

from threshold import fit_model, predict

correct = 0.0
for ei in range (len(features)):
    training = np.ones(len(features), bool)
    training[ei] = False
    testing = ~training
    model = fit_model(features[training], is_virginica[training])
    predictions = predict(model, features[testing])
    correct += np.sum(predictions == is_virginica[testing])
acc = correct/float(len(features))
print("Mean accuracy: {0:.1%}".format(acc))












































































































































































import numpy as np

def fit_model(features, labels):
    '''Learn a simple threshold model'''
    best_acc = -1.0
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)
            acc = (pred == labels).mean()
            rev_acc = (pred == ~labels).mean()
            if rev_acc > acc:
                acc = rev_acc
                reverse = True
            else:
                reverse = False

            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse

    return best_t, best_fi, best_reverse


def predict(model ,features):
    '''Apply a learned model'''
    t, fi, reverse = model
    if reverse:
        return features[:, fi] <= t
    else:
        return features[:, fi] > t

def accuracy(features, labels, model):
    '''Compute the accuracy of the model'''
    preds = predict(model, features)
    return np.mean(preds == labels)





























